# 必要なライブラリをインポート
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')
import datetime
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import japanize_matplotlib
%matplotlib inline


# 平均計算期間毎にプロットする
def SPG(company_code, start, end):
    # 移動平均を指定した全機関で算出するため、余分な期間も含めてスクレイピングする
    start = (datetime.datetime.strptime(start, '%Y-%m-%d')-datetime.timedelta(days=80)).strftime('%Y-%m-%d')
    df = pdr.StooqDailyReader(company_code, start=start, end=end).read().sort_index()
    
    price = df['Close']
    span01 = 5
    span02 = 25
    span03 = 50
    
    def wma(w):
        weight = np.arange(len(w)) + 1
        wma = np.sum(weight * w) / weight.sum()
        return wma
    
    df['sma01'] = price.rolling(window=span01).mean()
    df['sma02'] = price.rolling(window=span02).mean()
    df['sma03'] = price.rolling(window=span03).mean()
    
    df['wma01'] = price.rolling(window=span01).apply(wma,raw=True).round(1)
    df['wma02'] = price.rolling(window=span02).apply(wma,raw=True).round(1)
    df['wma03'] = price.rolling(window=span03).apply(wma,raw=True).round(1)
    
    df['ema01'] = price.ewm(span=span01, adjust=False).mean()
    df['ema02'] = price.ewm(span=span02, adjust=False).mean()
    df['ema03'] = price.ewm(span=span03, adjust=False).mean()
    
    start = (datetime.datetime.strptime(start, '%Y-%m-%d')+datetime.timedelta(days=80)).strftime('%Y-%m-%d')
    df = df[(start <= df.index) & (df.index <= end)]
    date = df.index
    price = df['Close']
    
    plt.figure(figsize=(30,15))
    
    # 5年
    plt.subplot(4,1,1)
    plt.title('5日',color='black',backgroundcolor='white',size=30,loc='center')
    plt.plot(date,price,label='Close',color='black')
    plt.plot(date,df['sma01'],label='sma',color='red')
    plt.plot(date,df['wma01'],label='wma',color='blue')
    plt.plot(date,df['ema01'],label='ema',color='green')
    plt.legend()
    
    # 25年
    plt.subplot(4,1,2)
    plt.title('25日',color='black',backgroundcolor='white',size=30,loc='center')
    plt.plot(date,price,label='Close',color='black')
    plt.plot(date,df['sma02'],label='sma',color='red')
    plt.plot(date,df['wma02'],label='wma',color='blue')
    plt.plot(date,df['ema02'],label='ema',color='green')
    plt.legend()
    
    # 50年
    plt.subplot(4,1,3)
    plt.title('50日',color='black',backgroundcolor='white',size=30,loc='center')
    plt.plot(date,price,label='Close',color='black')
    plt.plot(date,df['sma03'],label='sma',color='red')
    plt.plot(date,df['wma03'],label='wma',color='blue')
    plt.plot(date,df['ema03'],label='ema',color='green')
    plt.legend()
    
    # 取引量
    plt.subplot(4,1,4)
    plt.title('Volume',color='black',backgroundcolor='white',size=30,loc='center')
    plt.bar(date,df['Volume'],label='Volume',color='grey')
    plt.legend()


# テスト
# S&P 500：1547.jp
SPG('1547.jp', '2021-03-10', '2022-03-10')
