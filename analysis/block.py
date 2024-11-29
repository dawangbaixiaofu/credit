
import csv 
from tqdm import tqdm
from config_etc import MarketingDurationConfig, MarketingUsersConfig
from config_etc import ActionSQLConfig
from config_etc import DataConfig


spark = None


class Action:
    def __init__(self) -> None:
        pass

    def get_action(self):
        return spark.sql(ActionSQLConfig.sql)



class Features:
    def __init__(self, config_path:str = "./config_features.csv") -> None:
        self.config_path = config_path
        
    def feature_static(self, row:dict):
        assert row['type'] == "static", "feature's type must be 'static'"

        sql = f"""
        with users as (
            select distinct {MarketingUsersConfig.id_field} as ccif_no
            from {MarketingUsersConfig.table}
        )
        , feat as (
        select distinct {row.id_field} as ccif_no, {row.feature_field} as {row.customized_feature_name}
        from {row.table}
        where {row.datetime_field} = '{MarketingDurationConfig.start_ds}'
        )
        select distinct users.ccif_no, {row.customized_feature_name}
        from users 
        inner join feat 
        on users.ccif_no = feat.ccif_no
        """
        return spark.sql(sql)
    

    def feature_dynamic(self, row:dict):
        pass

    def feature_trend(self, row:dict):
        pass


    def get_features(self):
        feats = {}
        with open(self.config_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['type'] == 'static':
                    temp = self.feature_static(row)
                elif row['type'] == 'dynamic':
                    temp = self.feature_dynamic(row)
                elif row['type'] == 'trend':
                    temp = self.feature_trend(row)
                feats[row.customized_feature_name] = temp
        return feats


class Data:
    def __init__(self) -> None:
        self.data = None

    def gen_data(self):
        self.users_action = Action().get_action()
        self.users_features:dict = Features().get_features()
        for feat_name in tqdm(self.users_features.keys()):
            DF = self.users_features[feat_name]
            if self.data is None:
                self.data = self.users_action.join(other=DF, how='left', on='ccif_no')
            else:
                self.data = self.data.join(other=DF, how='left', on='ccif_no')
            # self.data = self.data.fillna(0)
            # print(feat_name)
        self.data.write.mode("overwrite").saveAsTable(DataConfig.output_table)
        return self.data


    def load(self):
        return spark.sql(f"""select * from {DataConfig.output_table}""")
        


if __name__ == "__main__":
    # analysis = Data()
    # analysis_data = analysis.data
    # analysis_data.shape
    print(ActionSQLConfig.sql)
    # from time import sleep
    # feats = {"withdraw": 2, 'credit':3, 'coupon':4, 'deposit':9}
    # for name in tqdm(feats.keys()):
    #     sleep(1)
    #     # print(name)


            