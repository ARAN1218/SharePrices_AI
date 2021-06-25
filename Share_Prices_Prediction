#必要なライブラリをインポート
from pandas_datareader import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import datetime


#AIの学習/予測のメインプログラム
#SharePricesPrediction = SPP
def SPP(start,end,company_code):
    #株価データをスクレイピングし、指定された範囲に前処理する。
    df = data.DataReader(company_code,"stooq")
    df = df[(df.index>=start) & (df.index<=end)]
    date = df.index
    price = df["Close"].to_numpy()
    df = df.drop('Close',axis=1)
    
    #重回帰分析のAIモデルを作成し、学習させる。
    model = LinearRegression()
    model.fit(df,price)
    
    #学習させたAIモデルの訓練データに対する精度を評価する。
    print('Train_score')
    print('MAE:',mean_absolute_error(price,model.predict(df)))
    print('MSE:',mean_squared_error(price,model.predict(df)))
    print('RMSE:',np.sqrt(mean_squared_error(price,model.predict(df))))
    print('R^2:',model.score(df,price))
    print(pd.DataFrame(model.coef_,columns=['回帰係数'],index=df.columns))
    
    #予測データをデータフレームに変換する。
    df_p = model.predict(df).reshape((len(model.predict(df)),1))
    df_p = pd.DataFrame(data=df_p,index=date,columns=['Close_p'])
    
    #実データと予測データをグラフにプロットする。
    plt.figure(figsize=(30,15))
    plt.subplot(2,1,1)
    plt.title('Train',color='black',backgroundcolor='white',size=30,loc='center')
    plt.plot(date,price,label='Close',color='red')
    plt.plot(date,df_p['Close_p'],label='Close_p',color='blue')
    plt.legend()
    
    #株価データを再度スクレイピングし、一年後の範囲で前処理する。
    start = datetime.datetime.strptime(start,'%Y-%m-%d')
    end = datetime.datetime.strptime(end,'%Y-%m-%d')
    df = data.DataReader(company_code,"stooq")
    df = df[(df.index>=start+datetime.timedelta(days=365)) & (df.index<=end+datetime.timedelta(days=365))]    
    date = df.index
    price = df['Close']
    df = df.drop('Close',axis=1)
    
    #テストデータに対する精度を評価する。
    print('Test_score')
    print('MAE:',mean_absolute_error(price,model.predict(df)))
    print('MSE:',mean_squared_error(price,model.predict(df)))
    print('RMSE:',np.sqrt(mean_squared_error(price,model.predict(df))))
    print('R^2',model.score(df,price))
    
    #予測データをデータフレームに変換する。
    df_p = model.predict(df).reshape((len(model.predict(df)),1))
    df_p = pd.DataFrame(data=df_p,index=date,columns=['Close_p'])
    
    #実データと予測データをグラフにプロットする。
    plt.subplot(2,1,2)
    plt.title('Test',color='black',backgroundcolor='white',size=30,loc='center')
    plt.plot(date,price,label='Close',color='red')
    plt.plot(date,df_p['Close_p'],label='Close_p',color='blue')
    plt.legend()
    
#SPP(株価データ収集期間の初めの年月日, 株価データ収集期間の終わりの年月日, 東京証券取引所等で調べた企業コード+.jp)
#今回は例として任天堂の企業コードを使用
SPP('2019-04-01','2020-04-01','7974.jp')
