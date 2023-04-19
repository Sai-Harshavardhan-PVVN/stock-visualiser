# Made with ❤️ by Harshavardhan PVVNS

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas_datareader.data as reader
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import plotly.express as px
import yfinance as yf


st.title("Stock Visualizer!!")

st.sidebar.title('Investor Dashboard')

@st.cache(persist=True)
def load_all_tickers():
    url = 'https://en.wikipedia.org/wiki/NIFTY_50'
    df1 = pd.read_html(url)
    df=df1[1]
    return df
df = load_all_tickers()


#grouping sector wise
df2=df.groupby('Sector')




#list of tickers grouped
#Automobile
Automobile1 = df2.get_group('Automobile')['Symbol']
Automobile = [0]*0
for i in range(0,6):
   Automobile.append(Automobile1.iloc[i])


#Banking
Bank = df2.get_group('Banking')['Symbol']
Banking = [0]*0
for i in range(0,6):
   Banking.append(Bank.iloc[i])

#Cement
Cem = df2.get_group('Cement')['Symbol']
Cement = [0]*0
for i in range(0,3):
  Cement.append(Cem.iloc[i])

#Consumer_Goods
tel = df2.get_group('Consumer Goods')['Symbol']
Consumer_Goods= [0]*0
for i in range(0,6):
 Consumer_Goods.append(tel.iloc[i])

#Energy
en = df2.get_group('Energy - Oil & Gas')['Symbol']
Energy = [0]*0
for i in range(0,4):
 Energy.append(en.iloc[i])

#Power
po = df2.get_group('Energy - Oil & Gas')['Symbol']
Power = [0]*0
for i in range(0,2):
 Power.append(en.iloc[i])

#Financial_Services
fin = df2.get_group('Financial Services')['Symbol']
Financial_Services = [0]*0
for i in range(0,5):
 Financial_Services.append(fin.iloc[i])

#Information_Technology
it = df2.get_group('Information Technology')['Symbol']
Information_Technology = [0]*0
for i in range(0,5):
 Information_Technology.append(it.iloc[i])

#Metals
met = df2.get_group('Metals')['Symbol']
Metals = [0]*0
for i in range(0,4):
 Metals.append(met.iloc[i])

#Pharmaceuticals
ph = df2.get_group('Pharmaceuticals')['Symbol']
Pharmaceuticals= [0]*0
for i in range(0,4):
 Pharmaceuticals.append(ph.iloc[i])

#Line_chart
 def Line_chart(df):
     st.line_chart(data=df)

#datetime
start = dt.date(2019,5,1)
end = dt.datetime.now()
titles = ['Open','High','Low','Close','Adj Close', 'Volume' ]

#interactive_candelsticks
def interactive_candelsticks(df1):
    fig = go.Figure(data =[go.Candlestick(x=df1.index,
                                     open=df1['Open'],
                                     high = df1['High'],
                                     low = df1['Low'],
                                     close = df1['Close'])])

    st.plotly_chart(fig)

# mpf plots
def mpf_plots(df):
    #
    fig=mpf.plot(df, type='candle', style='yahoo',
        title='Moving Averages',
        ylabel='Price',
        ylabel_lower='',
        volume=True,
        mav=(50,200),tight_layout=True,figscale=0.75

       )
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)
    #st.plt.legend()


#RSI
def get_RSI(df, column='Adj Close', time_window=14):
    """Return the RSI indicator for the specified time window."""
    diff = df[column].diff(1)

    # This preservers dimensions off diff values.
    up_chg = 0 * diff
    down_chg = 0 * diff

    # Up change is equal to the positive difference, otherwise equal to zero.
    up_chg[diff > 0] = diff[diff > 0]

    # Down change is equal to negative deifference, otherwise equal to zero.
    down_chg[diff < 0] = diff[diff < 0]

    # We set com = time_window-1 so we get decay alpha=1/time_window.
    up_chg_avg = up_chg.ewm(com=time_window - 1,
                            min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window - 1,
                                min_periods=time_window).mean()

    RS = abs(up_chg_avg / down_chg_avg)
    df['RSI'] = 100 - 100 / (1 + RS)

    return df

def plot_volume(fig, df, row, column=1):
    """Return a graph object figure containing the volume chart in the specified row."""
    fig.add_trace(go.Bar(x=df.index,
                         y=df['Volume'],
                         marker=dict(color='lightskyblue',
                                     line=dict(color='firebrick', width=0.1)),
                         showlegend=False,
                         name='Volume'),
                  row=row,
                  col=column)

    fig.update_xaxes(title_text='Date', row=row, col=column)
    fig.update_yaxes(title_text='Volume ', row=row, col=column)

    return fig

#rsi plotly
def plot_RSI(fig, df, row, column=1):
    """Return a graph object figure containing the RSI indicator in the specified row."""
    df1=df.reset_index()
    fig.add_trace(go.Scatter(x=df1['Date'].iloc[30:],
                             y=df1['RSI'].iloc[30:],
                             name='RSI',
                             line=dict(color='gold', width=2)),
                  row=row,
                  col=column)

    fig.update_yaxes(title_text='RSI', row=row, col=column)

    # Add one red horizontal line at 70% (overvalued) and green line at 30% (undervalued)
    for y_pos, color in zip([70, 30], ['Red', 'Green']):
        fig.add_shape(x0=df1['Date'].iloc[1],
                      x1=df1['Date'].iloc[-1],
                      y0=y_pos,
                      y1=y_pos,
                      type='line',
                      line=dict(color=color, width=2),
                      row=row,
                      col=column)

    # Add a text box for each line
    for y_pos, text, color in zip([64, 36], ['Overvalued', 'Undervalued'], ['Red', 'Green']):
        fig.add_annotation(x=df1['Date'].iloc[int(df1['Date'].shape[0] / 10)],
                           y=y_pos,
                           text=text,
                           font=dict(size=14, color=color),
                           bordercolor=color,
                           borderwidth=1,
                           borderpad=2,
                           bgcolor='lightsteelblue',
                           opacity=0.75,
                           showarrow=False,
                           row=row,
                           col=column)

    # Update the y-axis limits
    ymin = 25 if df1['RSI'].iloc[30:].min() > 25 else df1['RSI'].iloc[30:].min() - 5
    ymax = 75 if df1['RSI'].iloc[30:].max() < 75 else df1['RSI'].iloc[30:].max() + 5
    fig.update_yaxes(range=[ymin, ymax], row=row, col=column)

    return fig


def plot_candlestick_chart(fig, df, row, column=1):
    """Return a graph object figure containing a Candlestick chart in the specified row."""
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Candlestick Chart'),
                  row=row,
                  col=column)
    fig.update_xaxes(rangeslider={'visible': False})
    fig.update_yaxes(title_text='Price ', row=row, col=column)

    return fig



def relativeret(df):
    daily_returns = df.pct_change()
    daily_cum_returns = (daily_returns +1).cumprod()
    daily_cum_returns = daily_cum_returns.fillna(0)
    return daily_cum_returns

#peer_line_chart
def peer_line(sts):
    dropdown = st.multiselect('Pick your assest',sts)
    if len(dropdown)>0:
        df = relativeret(reader.get_data_yahoo(dropdown,start,end)['Adj Close'])
        #st.write(df)
        i=0
        fig = go.Figure()
        st.write(dropdown)
        for x in dropdown:
            fig.add_traces(go.Scatter(x=df.index,y=df[x],mode ='lines',name=x))

        st.plotly_chart(fig)



def bollinger_band(df):
    df['SMA'] = df.Close.rolling(window=20).mean()
    df['stddev']= df.Close.rolling(window=20).std()
    df['Upper']=df.SMA + 2*df.stddev
    df['Lower']=df.SMA - 2*df.stddev
    df['Buy_Signal']=np.where(df.Lower > df.Close,True,False)
    df['Sell_Signal']=np.where(df.Upper < df.Close,True,False)
    df=df.dropna()
    #st.write(df)
    #f#ig = plt.plot(df[['Close','SMA','Upper','Lower']])
    #st.pyplot(fig)
    dropdown1=['Close','SMA','Upper','Lower']
    fig1 = go.Figure()
    for x in dropdown1:
        fig1.add_traces(go.Scatter(x=df.index,y=df[x],mode ='lines',name=x))
    fig1.add_trace(go.Scatter(
    x=df['Upper'], y=df['Lower'],
    line_color='rgb(231,107,243)',
    name='Ideal',
))
    st.plotly_chart(fig1)



def bar_plot(df):
    st.bar_chart(df)



st.sidebar.subheader('Choose your intrested area')

#NIFTY_50
nifty = st.sidebar.checkbox("NIFTY_50",False,key='1')
if nifty:
    st.subheader("NIFTY_50")
    start = dt.date(2019,5,1)
    end = dt.datetime.now()
    ndf = reader.get_data_yahoo("^NSEI",start,end)
    titles = ['Open','High','Low','Close','Adj Close', 'Volume' ]
    ndf = ndf[titles]
    ndf1 = ndf['Close']
    st.markdown("## Line Chart")
    Line_chart(ndf1)
    st.write(" ")
    st.write(" ")
    st.markdown("## Range Slider Candlestick")
    interactive_candelsticks(ndf)
    st.markdown("## Moving averages 50 & 200")
    mpf_plots(ndf)
    ndf=get_RSI(ndf)
    #st.write(ndf)
    fig = make_subplots(rows=3,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.005,
                    row_width=[0.2, 0.3, 0.8])

    fig = plot_candlestick_chart(fig,
                             ndf,
                             row=1)
    #fig = plot_MACD(fig, df, row=2)
    fig = plot_RSI(fig, ndf, row=2)

    fig = plot_volume(fig, ndf, row=3)
    st.write("  ")
    st.markdown("## RSI with Volume")
    st.plotly_chart(fig)


# BAnk NIFTY_50
bank = st.sidebar.checkbox("Bank Nifty",False,key='2')
if bank:
    st.subheader("Bank Nifty")
    start = dt.date(2019,5,1)
    end = dt.datetime.now()
    ndf = reader.get_data_yahoo("^NSEBANK",start,end)
    titles = ['Open','High','Low','Close','Adj Close', 'Volume' ]
    ndf = ndf[titles]
    ndf1 = ndf['Close']
    st.markdown("## Line Chart")
    Line_chart(ndf1)
    st.write(" ")
    st.write(" ")
    st.markdown("## Range Slider Candlestick")
    interactive_candelsticks(ndf)
    st.markdown("## Moving averages 50 & 200")
    mpf_plots(ndf)
    ndf=get_RSI(ndf)

    fig = make_subplots(rows=3,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.005,
                    row_width=[0.2, 0.3, 0.8])

    fig = plot_candlestick_chart(fig,
                             ndf,
                             row=1)
    #fig = plot_MACD(fig, df, row=2)
    fig = plot_RSI(fig, ndf, row=2)

    fig = plot_volume(fig, ndf, row=3)
    st.write("  ")
    st.markdown("## RSI with Volume")
    st.plotly_chart(fig)


#sectors
dct={"Automobile":Automobile,"Banking":Banking,"Cement":Cement,"Consumer Goods":Consumer_Goods,"Energy - Oil & Gas":Power,"Financial Services":Financial_Services,"Information Technology":Information_Technology,"Metals":Metals,"Pharmaceuticals":Pharmaceuticals}
sectors=["None","Automobile","Banking","Cement","Consumer Goods","Energy - Oil & Gas","Financial Services","Information Technology","Metals","Pharmaceuticals"]
sector = st.sidebar.selectbox("Select your sector",sectors)

if sector is not "None":
    st.markdown(sector)
    st.sidebar.markdown("#### Select your type of Analysis")
    tech=st.sidebar.checkbox("Technicals",False,key="3")

    fund=st.sidebar.checkbox("Fundamentals",False,key="5")


    if tech:
        stock=st.selectbox("Select your Stock",dct[sector])
        st.markdown("## Relatives Closing prices Vs Peers")
        peer_line(dct[sector])
        ndf = reader.get_data_yahoo(stock,start,end)
        titles = ['Open','High','Low','Close','Adj Close', 'Volume' ]
        ndf = ndf[titles]
        ndf1 = ndf['Close']
        st.markdown("## Line Chart")
        Line_chart(ndf1)
        st.write(" ")
        st.write(" ")
        st.markdown("## Range Slider Candlestick")
        interactive_candelsticks(ndf)
        st.markdown("## Moving averages 50 & 200")
        mpf_plots(ndf)
        ndf=get_RSI(ndf)

        fig = make_subplots(rows=3,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.005,
                        row_width=[0.2, 0.3, 0.8])

        fig = plot_candlestick_chart(fig,
                                 ndf,
                                 row=1)
        #fig = plot_MACD(fig, df, row=2)
        fig = plot_RSI(fig, ndf, row=2)

        fig = plot_volume(fig, ndf, row=3)
        st.write("  ")
        st.markdown("## RSI with Volume")
        st.plotly_chart(fig)
        #

        #bollinger_band(ndf)




    if fund:
        dropdown2 = st.multiselect('Pick your assest',dct[sector],key='9')
        st.write(dropdown2)
        if len(dropdown2)>1:
            infos=[]
            for i in dropdown2:
                infos.append(yf.Ticker(i).info)
            fundament = ['marketCap','priceToBook','trailingEps','trailingPE']
            dff = pd.DataFrame(infos)
            dff = dff.set_index('symbol')
            dff1=dff[dff.columns[dff.columns.isin(fundament)]]
            dff1
            mkt=dff1.iloc[:,3]
            eps=dff1.iloc[:,0]
            ptb=dff1.iloc[:,1]
            pe=dff1.iloc[:,2]
            st.markdown("## Earnings per Share")
            bar_plot(eps)
            st.markdown("## Price To Book Ratio")
            bar_plot(ptb)
            st.markdown("## Trailing P/E")
            bar_plot(pe)
            st.markdown("## Market Cap")
            bar_plot(mkt)
