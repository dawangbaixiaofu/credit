from block import Data
from config_etc import ActionConfig
from config_etc import AnalysisConfig
import csv
import pandas as pd
from tqdm import tqdm
from time import sleep


class Correlation:
    '''
    因变量是连续值类型，通过其和自变量之间的相关系数来筛选相关联的特征
    不同类型的变量间相关关系前，需要对不同类型变量组合进行穷举：
    - （离散，离散）
        - （离散，二分类）
        - （离散，多分类）
            - （有序，有序）
            - （有序，无序）
            - （无序，无序）
    - （离散，连续）
        - （无序，连续）
        - （有序，连续）
    - （连续，连续）

    不同类型变量间的相关关系计算方式如下：
    - （离散型变量，binary category离散型变量）
        IV
        WOE
        TGI = 特定客群中具有某一特征的人群比例/所有客群中具有某一特征的人群比例
        

    - （离散型变量，multiple categories离散型变量） --> 转化为多个（离散， binary category_i)
        TGI
        - （有序，有序）
            Kendall相关系数。引入'一致对'和'分歧对'概念
            Spearman相关系数
        - （有序，无序）
            --> （无序，无序）
        - （无序，无序）
            卡方检验
            ANOVA(analysis of variance)
    
    - （离散型变量，连续性变量）
        method_1:
        input:
        user_j_feature_value_i, target_value_i --> feature_value_i, mean_i
        (mean_1, ..., mean_n)  -> mean, variance
        (mean_i - mean, ..., mean_n - mean)
        通过variance来衡量离散变量带来的影响，其值越大说明该变量影响越大；
        通过mean_n - mean来衡量离散变量的具体那个值（对应的客群）对目标变量影响较大；

        - （有序离散，连续）
            参考Spearman相关系数

        - （无序离散，连续）
            参考method_1
            or -->（无序离散，有序离散）
        
    - （连续性变量，连续性变量）
        Pearson相关系数：假设变量的正态分布和等方差
        Spearman秩相关：没有假设变量的正态分布和等方差。
        把两个连续变量分桶后，再计算Pearson相关系数；


    reference: 
        - https://stats.stackexchange.com/questions/108007/correlations-with-unordered-categorical-variables

    '''
    def __init__(self) -> None:
        self.data = Data().load()
        self.cols = self.data.columns

    def _discrete_discrete_tgi(self, var_names:dict):
        pass

    def _discrete_binary_iv(self, var_names:dict):
        pass

    def _discrete_binary_woe(self, var_names:dict):
        pass

    
    def _discrete_continuous(self, var_names:dict):
        assert var_names['x'] in self.cols, f"'{var_names['x']}' not in self.data's columns."
        assert var_names['y'] in self.cols, f"'{var_names['y']}' not in self.data's columns."

        df:pd.DataFrame = self.data.select(*var_names.values()).toPandas()
        group = df.groupby(var_names['x']).mean().reset_index()
        
        variance:float = group[var_names['y']].var()
        avg:float = df[var_names['y']].mean()
        group['diff'] = group[var_names['y']] - avg
        group = group.sort_values(by='diff')

        return variance, group[[var_names['x'],'diff']]
        


    def discrete_continuous(self):
        variances = {}
        diffs = {}
        with open(file="./config_features.csv.demo", mode='r') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader):
                var_names = {'x': row.customized_feature_name, 'y':ActionConfig.customized_action_name}
                variance, diff = self._discrete_continues(var_names)
                variances[row.customized_feature_name] = variance
                diffs[row.customized_feature_name] = diff
        return variances, diffs
    
    
    def discrete_binary(self):
        # todo: 
        pass


class AutoFind:
    def __init__(self) -> None:
        pass


    def top_k(self, n:int=5, method:str = 'discrete_continuous'):
        '''
        按照给定的相关系数值进行排序选取top n
        method: 
          'discrete_continuous' （离散变量，连续变量）组间方差，组间均值差
          'discrete_binary' （离散变量，二分类变量）IV，WOE
          ...
        '''
        check_method_names = set(['discrete_continuous', 'discrete_binary'])

        assert method in check_method_names, f"analysis method name: '{method}' is invalid."

        corr = Correlation()
        if method == 'discrete_continuous':
            variances, diffs = getattr(corr, method)()
            ordered_top_variances = dict(sorted(variances.items(), key=lambda item:item[1])[:n])
            top_diffs = {key:value for key, value in diffs.items() if key in ordered_top_variances}
            # make analysis reportment
            with open(AnalysisConfig.analysis_reportment_file, mode='w+') as f:
                print(f"Top {n} 的特征如下：", file=f)
                print(variances, file=f)
                print("每个特征取值对应客群的相关程度：", file=f)
                for key in variances:
                    print(f"-----------{key}--------------", file=f)
                    print(top_diffs[key], file=f)
        
        # todo: 
        if method == 'discrete_binary':
            pass

        return
    

if __name__ == "__main__":
    ana = AutoFind()
    ana.top_k(n=5, method='discrete_continuous')
