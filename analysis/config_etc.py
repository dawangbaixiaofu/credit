class MarketingUsersConfig:
    table = "xxxxxxx"
    id_field = "xxx"


class MarketingDurationConfig:
    start_ds = "20241001"
    start_date = "2024-10-01"
    end_date = "2024-10-31"


class ActionConfig:
    action_name = "withdraw_amount"
    table = "xxxxxxxx"
    id_field = "xxx"
    datetime_field = "xxx"
    action_field = None
    customized_action_name = "action_label"

    if action_field is None:
        action_field = 1  # 如果action_filed是空的，比如是否提款，可以指定一个值




class ActionSQLConfig:
    sql = f"""
    with users as (
        select distinct {MarketingUsersConfig.id_field} as ccif_no
        from {MarketingUsersConfig.table}
    
    )
    , actions as (
        select distinct {ActionConfig.id_field} as ccif_no, {ActionConfig.datetime_field} as action_datetime, {ActionConfig.action_field}
        from {ActionConfig.table}
        where 
            {ActionConfig.datetime_field} >= '{MarketingDurationConfig.start_date}'
        and {ActionConfig.datetime_field} <= '{MarketingDurationConfig.end_date}'
        
        -- etc

    )
    select users.ccif_no, action_datetime, 
        case when {ActionConfig.action_field} is null then 0 
            else {ActionConfig.action_field}
            end as {ActionConfig.customized_action_name}
    from users 
    left join actions 
    on users.ccif_no = actions.ccif_no
    """

class DataConfig:
    output_table = ""


class AnalysisConfig:
    analysis_reportment_file = "./analysis_reportment.txt"
    

if __name__ == "__main__":
    print(ActionSQLConfig.sql)