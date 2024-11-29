from config_etc import MarketingUsersConfig
from config_etc import ActionConfig


class RepresentationConfig:
    # 展示时间区间配置
    start_date = "2024-10-01"
    end_date = "2024-10-31"

    # 同比时间区间配置
    compare_start_date = "2023-10-01"
    compare_end_date = "2023-10-31"
    
    # 统计周期粒度配置
    period_2_1 = "day"
    period_hierarchy = ['month', 'day']

    # 统计指标配置
    target = "mean"
    
    # 环比，同比配置
    compare_with_previous_s_2_1 = True
    compare_with_previous_s_2_2 = True
    compare_with_history_s_2_1 = True
    compare_with_history_s_2_2 = True

    # 其他配置...
    

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
            {ActionConfig.datetime_field} >= '{RepresentationConfig.compare_start_date}'
        and {ActionConfig.datetime_field} <= '{RepresentationConfig.end_date}'
        
        -- etc

    )
    select distinct users.ccif_no, action_datetime, 
        case when {ActionConfig.action_field} is null then 0 
            else {ActionConfig.action_field}
            end as {ActionConfig.customized_action_name}
    from users 
    inner join actions 
    on users.ccif_no = actions.ccif_no
    """



if __name__ == "__main__":
    print(ActionSQLConfig.sql)