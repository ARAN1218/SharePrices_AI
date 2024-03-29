#必要なライブラリをインポート
from pandas_datareader import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


#加重移動平均を計算する関数
def wma(w):
    weight = np.arange(len(w)) + 1
    wma = np.sum(weight * w) / weight.sum()
    return wma


#指数移動平均を計算する関数
def ema(e,p):
    ema = np.zeros(len(e))
    ema[:] = np.nan
    ema[p-1] = e[:p].mean()
    
    for d in range(p,len(e)):
        ema[d] = ema[d-1] + (e[d] - e[d-1]) / (p+1) * 2
    return ema


#株価データ自動取得のメインプログラム
#SharePricesGraphing = SPG
def SPG(start,end,company_code):
    #stooqから企業コードの株価データを取得し、テーブルデータを作成する
    df = data.DataReader(company_code,"stooq")
    df = df[(df.index>=start) & (df.index<=end)]
    
    #グラフの軸要素を取得
    date = df.index
    price = df['Close']
    
    #移動平均の期間を設定
    span01 = 5
    span02 = 25
    span03 = 50
    
    #移動平均を計算
    df['sma01'] = price.rolling(window=span01).mean()
    df['sma02'] = price.rolling(window=span02).mean()
    df['sma03'] = price.rolling(window=span03).mean()
    
    #加重移動平均を計算
    df['wma01'] = price.rolling(window=span01).apply(wma,raw=True).round(1)
    df['wma02'] = price.rolling(window=span02).apply(wma,raw=True).round(1)
    df['wma03'] = price.rolling(window=span03).apply(wma,raw=True).round(1)
    
    #指数移動平均を計算
    df['ema01'] = ema(price,span01).round(1)
    df['ema02'] = ema(price,span02).round(1)
    df['ema03'] = ema(price,span03).round(1)
    
    #グラフの大きさを調整
    plt.figure(figsize=(30,15))
    
    #移動平均をプロット
    plt.subplot(4,1,1)
    plt.title('SMA',color='black',backgroundcolor='white',size=30,loc='center')
    plt.plot(date,price,label='Close',color='black')
    plt.plot(date,df['sma01'],label='sma01',color='red')
    plt.plot(date,df['sma02'],label='sma02',color='blue')
    plt.plot(date,df['sma03'],label='sma03',color='green')
    plt.legend()
    
    #加重移動平均をプロット
    plt.subplot(4,1,2)
    plt.title('WMA',color='black',backgroundcolor='white',size=30,loc='center')
    plt.plot(date,price,label='Close',color='black')
    plt.plot(date,df['wma01'],label='wma01',color='red')
    plt.plot(date,df['wma02'],label='wma02',color='blue')
    plt.plot(date,df['wma03'],label='wma03',color='green')
    plt.legend()
    
    #指数移動平均をプロット
    plt.subplot(4,1,3)
    plt.title('EMA',color='black',backgroundcolor='white',size=30,loc='center')
    plt.plot(date,price,label='Close',color='black')
    plt.plot(date,df['ema01'],label='ema01',color='red')
    plt.plot(date,df['ema02'],label='ema02',color='blue')
    plt.plot(date,df['ema03'],label='ema03',color='green')
    #print(df['ema01'],df['ema02'],df['ema03'])
    plt.legend()
    
    #取引量をプロット
    plt.subplot(4,1,4)
    plt.title('Volume',color='black',backgroundcolor='white',size=30,loc='center')
    plt.bar(date,df['Volume'],label='Volume',color='grey')
    plt.legend()
    
    
#SPG(株価データ収集期間の初めの年月日, 株価データ収集期間の終わりの年月日, 東京証券取引所等で調べた企業コード+.jp)
#今回は例として任天堂の企業コードを使用
SPG('2019-04-01','2021-04-01','7974.jp')
