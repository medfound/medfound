#!/usr/bin/env python3
from pathlib import Path
import fire
import pandas as pd
from tqdm import tqdm
from medfound.utils import extract_response_diag, get_code_mapper


class DataTools(object):
    def __init__(self, num_workers=0, node=None):
        super(DataTools, self).__init__()
        if num_workers is None:
            num_workers = 0
        elif num_workers == 0:
            num_workers = 1
        elif num_workers < 0:
            num_workers = -1
        self._num_workers = num_workers
        self._node = node

    def response_to_preference_dataset(self, path_data_ori, generated_dir,
                                   score_dir, output_path,
                                   mapper_ref_path, mapper_model_path,
                                   seed=42):
        """Process preference dataset.

        Args:
            path_data_ori (str): Directory of original data.
            generated_dir (str): Directory to store the generated data.
            score_dir (str): Directory of the predicted score data.
            output_path (str): Directory or path to the output dataset
            mapper_ref_path (str): ICD map file path
            mapper_model_path (str): Model path
            seed (int, optional): Random seed. Defaults to 42.

        Raises:
            ValueError: Invalid directory of original data
        """
        if ',' in path_data_ori:
            paths_data_ori = path_data_ori.split(',')
        elif isinstance(path_data_ori, list) or \
                isinstance(path_data_ori, tuple):
            paths_data_ori = list(path_data_ori)
        elif isinstance(path_data_ori, str):
            paths_data_ori = [path_data_ori]
        else:
            raise ValueError('invalid path_data_ori')

        dfs_temp_all = []
        for path_data_ori in paths_data_ori:
            df_temp = pd.read_json(path_data_ori, lines=True)
            dfs_temp_all += [df_temp]
        df_data_ori = pd.concat(dfs_temp_all)
        df_data_ori = df_data_ori.reset_index(drop=True)

        generated_dir = Path(generated_dir)
        paths = sorted(generated_dir.glob('*.jsonl'))
        dfs_temp_all = []
        for path in tqdm(paths):
            path = Path(path)
            df_temp = pd.read_json(path, lines=True)
            dfs_temp_all += [df_temp]
        df_gen = pd.concat(dfs_temp_all)
        df_gen = df_gen.reset_index(drop=True)

        score_dir = Path(score_dir)
        paths = sorted(score_dir.glob('*.jsonl'))
        dfs_temp_all = []
        for path in tqdm(paths):
            path = Path(path)
            df_temp = pd.read_json(path, lines=True)
            dfs_temp_all += [df_temp]
        df_score = pd.concat(dfs_temp_all)
        df_score = df_score.reset_index(drop=True)

        mapper = get_code_mapper(mapper_ref_path, mapper_model_path)
        df_data = df_gen
        df_data = df_data.merge(df_score.set_index(['qid', 'i'])[['score']]
                                .rename(columns={'score': 'help_score'}),
                                left_on=['qid', 'i'],
                                right_index=True,
                                how='left')
        df_data['pred'] = df_data['response'].apply(extract_response_diag)
        df_data['label_code'] = mapper(df_data['label'].values)
        df_data['pred_code'] = mapper(df_data['pred'].values)

        df_pred = df_data
        col_pred_code = 'pred_code'
        col_true_code = 'label_code'
        col_code_score = 'code_score'
        col_help_score = 'help_score'
        code_shifts = [1, 3, 5, 7]
        df_pred = df_pred[df_pred['response_token_len'] != 0]
        for i, shift in enumerate(code_shifts):
            df_pred[f'match_{i}'] = (df_pred[col_true_code].str[:shift]
                                     == df_pred[col_pred_code].str[:shift])
        df_pred[col_code_score] = \
            df_pred[[f'match_{i}' for i in range(4)]].sum(axis=1)

        df_pred_sample = df_pred
        df_pred_sample = df_pred_sample.sort_values(col_code_score,
                                                    ascending=False)
        df_pred_sample = \
            df_pred_sample.drop_duplicates(['qid', col_code_score])
        count = df_pred_sample['qid'].value_counts()
        df_pred_sample1 = \
            df_pred_sample[df_pred_sample['qid'].isin(count[count > 1].index)]
        df_pred_sample1 = df_pred_sample1.groupby('qid').head(2)
        df_pred_sample = df_pred[df_pred['qid'].isin(count[count == 1].index)]
        df_pred_sample = \
            df_pred_sample[df_pred_sample[col_code_score] == len(code_shifts)]
        df_pred_sample2 = df_pred_sample.groupby('qid').head(2)
        df_pred_sample = df_pred[df_pred['qid'].isin(count[count == 1].index)]
        df_pred_sample = \
            df_pred_sample[df_pred_sample[col_code_score] < len(code_shifts)]
        df_pred_sample3 = df_pred_sample.groupby('qid').head(2)
        df_pred_sample3_1 = df_pred_sample3.groupby('qid').head(1)
        df_pred_sample3_2 = df_pred_sample3.groupby('qid').tail(1)
        df_pred_sample3_2 = df_pred_sample3_2.copy()
        df_pred_sample3_2['response'] = df_pred_sample3_2['qid']\
            .map(df_data_ori.set_index('qid')['response'])
        df_pred_sample3_2[col_code_score] = len(code_shifts) + 1
        df_pred_sample3_2[col_help_score] = 1
        df_pred_sample3 = pd.concat([df_pred_sample3_1, df_pred_sample3_2])
        df_pred_sample = pd.concat([
            df_pred_sample1,
            df_pred_sample2,
            df_pred_sample3,
        ])
        df_pred_sample = df_pred_sample.sample(frac=1, random_state=seed)
        df_pred_sample = df_pred_sample.\
            sort_values(['qid', col_code_score, col_help_score],
                        ascending=False)

        df_output = df_data_ori.copy()
        df_output['accepted'] = df_output['qid'].map(
            df_pred_sample.groupby('qid').head(1).set_index('qid')['response'])
        df_output['rejected'] = df_output['qid'].map(
            df_pred_sample.groupby('qid').tail(1).set_index('qid')['response'])
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = df_output
        data.to_json(
            output_path, orient='records', lines=True, force_ascii=False)

    def response_bootstrapping(self, generated_dir, generated_hint_dir,
                           output_path, mapper_ref_path, mapper_model_path,
                           seed=42):
        """Process data generated with bootstrapping.

        Args:
            generated_dir (str): Directory to store the generated data.
            generated_hint_dir (str): Directory to store the generated data
                with hint prompting.
            output_path (str): Path to the output dataset.
            mapper_ref_path (str): ICD map file path.
            mapper_model_path (str): Model path.
            seed (int, optional): Random seed. Defaults to 42.
        """
        generated_dir = Path(generated_dir)
        paths = sorted(generated_dir.glob('*.jsonl'))
        dfs_temp_all = []
        for path in tqdm(paths):
            path = Path(path)
            df_temp = pd.read_json(path, lines=True)
            dfs_temp_all += [df_temp]
        df_gen = pd.concat(dfs_temp_all)
        df_gen = df_gen.reset_index(drop=True)
        df_gen['score_response'] = 1

        generated_hint_dir = Path(generated_hint_dir)
        paths = sorted(generated_hint_dir.glob('*.jsonl'))
        dfs_temp_all = []
        for path in tqdm(paths):
            path = Path(path)
            df_temp = pd.read_json(path, lines=True)
            dfs_temp_all += [df_temp]
        df_gen_hinted = pd.concat(dfs_temp_all)
        df_gen_hinted = df_gen_hinted.reset_index(drop=True)
        df_gen_hinted['score_response'] = 0

        mapper = get_code_mapper(mapper_ref_path, mapper_model_path)
        df_data = pd.concat([df_gen, df_gen_hinted])
        df_data['pred'] = df_data['response'].apply(extract_response_diag)
        df_data['label_code'] = mapper(df_data['label'].values)
        df_data['pred_code'] = mapper(df_data['pred'].values)

        df_pred = df_data
        col_true_code = 'label_code'
        col_pred_code = 'pred_code'
        col_code_score = 'code_score'
        df_pred = df_pred[df_pred['response_token_len'] != 0]
        code_shifts = [1, 3, 5, 7]
        for i, shift in enumerate(code_shifts):
            df_pred[f'match_{i}'] = (df_pred[col_true_code].str[:shift] ==
                                     df_pred[col_pred_code].str[:shift])
        df_pred[col_code_score] = \
            df_pred[[f'match_{i}'
                     for i in range(len(code_shifts))]].sum(axis=1)

        df_pred_sample = df_pred
        df_pred_sample = df_pred_sample.sample(frac=1, random_state=seed)
        df_pred_sample = df_pred_sample.sort_values(
            ['qid', col_code_score, 'score_response'], ascending=False)

        df_output = df_pred_sample.copy()
        df_output = df_output.groupby('qid').head(1)
        df_output['text'] = df_output.apply(
            lambda x: '[[SEP]]'.join(x['text'].split('[[SEP]]')[:-1] +
                                     [x['response']]), axis=1)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = df_output
        data.to_json(output_path, orient='records',
                     lines=True, force_ascii=False)


if __name__ == '__main__':
    fire.Fire(DataTools)
