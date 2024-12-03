from config_etc import ActionConfig
from config_anormal import ActionSQLConfig
from config_etc import MarketingUsersConfig
from config_anormal import RepresentationConfig
import pandas as pd
from datetime import datetime


spark = None

class MarketingUserAction:
    def __init__(self) -> None:
        pass

    def get_marketing_users(self):
        sql = f"""
        select distinct {MarketingUsersConfig.id_field} as ccif_no
        from {MarketingUsersConfig.table}
        """
        return spark.sql(sql)
    

    def get_action(self):
        return spark.sql(ActionSQLConfig.sql)
    

class Statistic:
    def __init__(self) -> None:
        obj = MarketingUserAction()
        self.marketing_users:pd.DataFrame = obj.get_marketing_users().toPandas()
        # statistic_1
        self.marketing_users_cnt:int = self.marketing_users.shape[0]
        
        self.action:pd.DataFrame = obj.get_action().toPandas()
        # add year season month week day fields
        self.action['action_date'] = pd.to_datetime(self.action['action_date'])
        self.action['year'] = self.action['action_date'].map(lambda x: x.year)
        self.action['season'] = self.action['action_date'].map(lambda x: (x.month-1)//3+1)
        self.action['month'] = self.action['action_date'].map(lambda x: x.month)
        self.action['week'] = self.action['action_date'].map(lambda x: x.isocalendar()[1])
        self.action['day'] = self.action['action_date'].map(lambda x: x.day)
        
    def set_duration(self, type:str="show"):
        if type=='show':
            start_date = pd.to_datetime(RepresentationConfig.start_date)
            end_date = pd.to_datetime(RepresentationConfig.end_date)
        elif type == 'compare':
            start_date = pd.to_datetime(RepresentationConfig.compare_start_date)
            end_date = pd.to_datetime(RepresentationConfig.compare_end_date)
        else:
            raise TypeError(f"this type '{type}' is invalid.")
        action = self.action[(self.action['action_date']>= start_date) & (self.action['action_date'] <= end_date)]
        return action


    def statistic_2_1(self, type:str="show") -> pd.DataFrame:
        period:str = RepresentationConfig.period_2_1
        target:str = RepresentationConfig.target


        periods = ['year', 'season', 'month', 'week', 'day']
        index = periods.index(period)
        by = periods[:index+1]

        action = self.set_duration(type=type)

        # todo: ./anormal_analysis.drawio::statistic::issue:approximation
        if target in ["sum", "count", "mean"]:
            s = action.groupby(by=by).agg({f"{ActionConfig.customized_action_name}": target})
        
        # todo: 人产，百产
        elif target == "production_per_user":
            pass
        elif target == "production_per_hundred_users":
            pass
        elif target == "mean_by_user":
            # calculate sum, calculate distinct count, then calculate mean by distinct user
            s = action.groupby(by=by).agg({ActionConfig.customized_action_name:"sum"})
            cnt:pd.Series = action.groupby(by=by)['ccif_no'].nunique()
            s[ActionConfig.customized_action_name] = s[ActionConfig.customized_action_name]/cnt
        else:
            raise TypeError(f"argument target '{target}' is invalid.")
        self.s_2_1 = s.reset_index()
        return self.s_2_1
    
    def statistic_2_2(self, type:str="show") -> pd.DataFrame:
        period_hierarchy:list = RepresentationConfig.period_hierarchy
        target:str = RepresentationConfig.target

        assert len(period_hierarchy) == 2, f"the length of argument period_hierarchy must be equal to 2. but got {len(period_hierarchy)}."

        period_2_1, period_2_2 = period_hierarchy

        periods = ['year', 'season', 'month', 'week', 'day']
        index = periods.index(period_2_2)
        by = periods[:index+1]

        s_2_1 = self.statistic_2_1(type=type)

        if target in ["sum", "count", "mean", "mean_by_user", "production_per_user", "production_per_hundred_users"]:
            s = s_2_1.groupby(by=by).agg({f"{ActionConfig.customized_action_name}": "mean"})
            
        else:
            raise TypeError(f"argument target '{target}' is invalid.")
        
        self.s_2_2 = s.reset_index()
        return self.s_2_2
    

    def compare_with_previous(self, target_type:str = "s_2_1") -> pd.DataFrame:
        """
        环比
        """
        if target_type == "s_2_1":
            cur_data = self.statistic_2_1(type='show')
        elif target_type == "s_2_2":
            cur_data = self.statistic_2_2(type='show')
        else:
            raise TypeError(f"target type '{target_type}' is invalid.")
        assert cur_data.shape[0] >= 2, f"length of cur_data is {cur_data.shape[0]}, which is less than 2. It is invalid for sequential compare."
        compare_data = cur_data[:-1]
        compare_data.reset_index(drop=True, inplace=True)

        cur_data = cur_data[1:]
        cur_data.reset_index(drop=True, inplace=True)
        return cur_data, compare_data


    def compare_with_history(self, target_type:str = "s_2_1") -> tuple:
        """
        同比
        """
        if target_type == "s_2_1":
            cur_data = self.statistic_2_1(type='show')
            compare_data = self.statistic_2_1(type='compare')
        
        elif target_type == "s_2_2":
            cur_data = self.statistic_2_2(type='show')
            compare_data = self.statistic_2_2(type='compare')
        else:
            raise TypeError(f"target type {target_type} is invalid.")
        
        # 历史对比的时候，不同年份的相同月份对应的周可能不一致
        min_cnt = min(cur_data.shape[0], compare_data.shape[0])
        cur_data = cur_data[:min_cnt]
        compare_data = compare_data[:min_cnt]
        return cur_data, compare_data

    




if __name__ == "__main__":
    dt = datetime(2024, 11, 27)
    # pd.to_datetime()
    print(dt,dt.day)
    print(dt.isocalendar())
    df = pd.DataFrame()
    df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
    'category_1':['A', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'D'],
    'category_2':['A', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'D'],
    'Values': [10, 20, 10, 40, 50, 60, 70, 80, 90, 100]
    })
    s1 = df.groupby(['Category', 'category_1', 'category_2']).agg({'Values':'sum'})
    print(s1)
    s2 = df.groupby(['Category', 'category_1', 'category_2'])['Values'].nunique()
    print(s2)
    s3 = df.groupby(['Category', 'category_1', 'category_2']).agg({'Values':'count'})
    print(s3)
    s1['Values'] = s1['Values']/s2
    print(s1)