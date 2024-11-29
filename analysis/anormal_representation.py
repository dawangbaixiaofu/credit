from anormal_data import Statistic
from config_anormal import RepresentationConfig
from config_etc import ActionConfig
import matplotlib.pyplot as plt
import pandas as pd

class Representation:
    """
    画图展示
    - 按需计算，根据需要画的图形，计算对应的数据；
    - 图形最小展示单元；
    - 图形组合，能够组合的最小图形，进行组合展示；
    """
    def __init__(self) -> None:
        self.statis_obj = Statistic()
        

    def compute_signal(self, method:str, kwargs:dict):
        return getattr(self.statis_obj, method)(**kwargs)


    @staticmethod
    def gen_x_ticks(row:pd.Series, period:str):
        periods = ['year', 'season', 'month', 'week', 'day']
        index = periods.index(period)
        by = periods[:index+1]
        res = [str(row[col]) for col in by] # concate year,season,month...
        return "-".join(res)
    
    @staticmethod
    def plot_trend_figure(data):
        fig, ax = plt.subplots(1,1)
        ax.plot(data['x'], data[ActionConfig.customized_action_name])
        ax.set_xlabel("date")
        ax.set_ylabel(f"{RepresentationConfig.target}")
        ax.set_title(f"{ActionConfig.customized_action_name}_{RepresentationConfig.target}")
        return fig, ax


    def trend_figure(self):
        if RepresentationConfig.period_2_1 is not None:
            data = self.compute_signal(method="statistic_2_1", kwargs={'type':"show"})        
            period = RepresentationConfig.period_2_1
            data['x'] =  data.apply(Representation.gen_x_ticks, axis = 1, period=period)

            Representation.plot_trend_figure(data)
        
        if RepresentationConfig.period_hierarchy is not None:
            data = self.compute_signal(method="statistic_2_2", kwargs={'type':"show"})
            _, period = RepresentationConfig.period_hierarchy
            data['x'] =  data.apply(Representation.gen_x_ticks, axis = 1, period=period)

            Representation.plot_trend_figure(data)
        return


    @staticmethod
    def plot_compare_figure(cur_data, compare_data):
        fig, axs = plt.subplots(1,2) # 2个子图，一个子图画当前指标和对比指标图，一个子图画环比图，
        axs[0].plot(cur_data['x'], cur_data[ActionConfig.customized_action_name], label='current')
        axs[0].plot(cur_data['x'], compare_data[ActionConfig.customized_action_name], label='compare')
        axs[0].set_xlabel("date")
        axs[0].set_ylabel(f"{RepresentationConfig.target}")
        axs[0].set_title(f"{ActionConfig.customized_action_name}_{RepresentationConfig.target}")
        plt.legend()

        axs[1].plot(cur_data['x'], cur_data['ratio'])
        axs[1].set_xlabel("date")
        axs[1].set_ylabel(f"{RepresentationConfig.target}_ratio")
        axs[1].set_title(f"{ActionConfig.customized_action_name}_{RepresentationConfig.target}_compare_increase_ratio")
        plt.show()
        return fig, axs


    def compare_figure(self):
        if RepresentationConfig.compare_with_previous_s_2_1:
            cur_data, compare_data = self.compute_signal(method='compare_with_previous', kwargs={"target_type":"s_2_1"})
            period = RepresentationConfig.period_2_1
            cur_data['x'] = cur_data.apply(Representation.gen_x_ticks, axis=1, period=period)
            cur_data['ratio'] = cur_data[ActionConfig.customized_action_name] / compare_data[ActionConfig.customized_action_name] - 1

            Representation.plot_compare_figure(cur_data, compare_data)

        if RepresentationConfig.compare_with_previous_s_2_2:
            cur_data, compare_data = self.compute_signal(method='compare_with_previous', kwargs={"target_type":"s_2_2"})
            _, period = RepresentationConfig.period_hierarchy
            cur_data['x'] = cur_data.apply(Representation.gen_x_ticks, axis=1, period=period)
            cur_data['ratio'] = cur_data[ActionConfig.customized_action_name] / compare_data[ActionConfig.customized_action_name] - 1
            
            Representation.plot_compare_figure(cur_data, compare_data)
        

        if RepresentationConfig.compare_with_history_s_2_1:
            cur_data, compare_data = self.compute_signal(method='compare_with_history', kwargs={"target_type":"s_2_1"})
            period = RepresentationConfig.period_2_1
            cur_data['x'] = cur_data.apply(Representation.gen_x_ticks, axis=1, period=period)
            cur_data['ratio'] = cur_data[ActionConfig.customized_action_name] / compare_data[ActionConfig.customized_action_name] - 1

            Representation.plot_compare_figure(cur_data, compare_data)

        if RepresentationConfig.compare_with_history_s_2_2:
            cur_data, compare_data = self.compute_signal(method='compare_with_history', kwargs={"target_type":"s_2_2"})
            _, period = RepresentationConfig.period_hierarchy
            cur_data['x'] = cur_data.apply(Representation.gen_x_ticks, axis=1, period=period)
            cur_data['ratio'] = cur_data[ActionConfig.customized_action_name] / compare_data[ActionConfig.customized_action_name] - 1

            Representation.plot_compare_figure(cur_data, compare_data)

        return


if __name__ == "__main__":
    brush = Representation()
    brush.trend_figure()
    brush.compare_figure()
    