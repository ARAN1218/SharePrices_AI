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


# 平均を丸める期間を自分で選択できるタイプ
# 引数spanに平均を計算する期間を入力する
def SPG(company_code, start, end, span=1):
    # 平均算出のため、指定範囲より多めにデータを取っておく
    start = (datetime.datetime.strptime(start, '%Y-%m-%d')-datetime.timedelta(days=80)).strftime('%Y-%m-%d')
    df = pdr.StooqDailyReader(company_code, start=start, end=end).read().sort_index()
    
    # 各種平均計算
    date = df.index
    price = df['Close']
    
    def wma(w):
        weight = np.arange(len(w)) + 1
        wma = np.sum(weight * w) / weight.sum()
        return wma
    
    df['sma'] = price.rolling(window=span).mean()
    df['wma'] = price.rolling(window=span).apply(wma,raw=True).round(1)
    df['ema'] = price.ewm(span=span, adjust=False).mean()
    
    # 指定された期間に範囲を切り取る
    start = (datetime.datetime.strptime(start, '%Y-%m-%d')+datetime.timedelta(days=80)).strftime('%Y-%m-%d')
    df = df[(start <= df.index) & (df.index <= end)]
    date = df.index
    price = df['Close']
    
    plt.figure(figsize=(30,15))
    
    # 指定された丸め期間で計算された各種平均をプロット
    plt.subplot(2,1,1)
    plt.title(f'{span}日',color='black',backgroundcolor='white',size=30,loc='center')
    plt.plot(date,price,label='Close',color='black')
    plt.plot(date,df['sma'],label='sma',color='red')
    plt.plot(date,df['wma'],label='wma',color='blue')
    plt.plot(date,df['ema'],label='ema',color='green')
    plt.legend()
    
    # 取引量
    plt.subplot(2,1,2)
    plt.title('Volume',color='black',backgroundcolor='white',size=30,loc='center')
    plt.bar(date,df['Volume'],label='Volume',color='grey')
    plt.legend()


# テスト
# S&P 500：1547.jp
SPG('1547.jp', '2021-03-10', '2022-03-10', span=10)
