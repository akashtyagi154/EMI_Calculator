import streamlit as st
import pandas as pd
import numpy as np
import math
# import plotly.express as px


## emi(p,r,t)
#     p: principal loan amount, r: rate of interest, t: time period for loan
#         returns monthly emi amount for given loan details

def emi(P = 0,R=0,T=0):
    P= float(P)
    R= float(R)
    T= float(T)
    R = R/1200
    T = T*12
    emi = round(P * R * ((math.pow((1+R),T))/(math.pow((1+R),T)-1)),2)
#     print(f'\n For Loan Amount: Rs. {P}/-, Interest rate: {round(R*1200,2)}%, Loan Tenure: {round(T/12)} Years  \n Total payable amount: Rs. {round(emi * T,2)}/- \n Monthly EMI: Rs. {emi} per month for {round(T)} months')
    return emi

## max_loan_amount_on_capped_emi(emi_cap,R,T)
#     emi_cap: maximum emi affordable by user, r: rate of interest, t: time period for loan
#     returns maximum principal loan amount for given emi amount and loan details


def max_loan_amount_on_capped_emi(emi_cap = 0,R=0,T=0):
    R= float(R)
    T= float(T)
    R = R/1200
    T = T*12
    return round(float(emi_cap)/ (R * (math.pow((1+R),T))/(math.pow((1+R),T)-1)),2)


## generate_monthly_installment_plan(loan_amount,rate,tenure)
#     loan_amount: principal loan amount, rate: rate of interest, tenure: time period for loan
#         returns monthly installment sheet for given loan details


def generate_monthly_installment_plan(loan_amount = 0,rate = 0,tenure = 0):
    emi_amount = emi(loan_amount, rate,tenure)
    main_list = [['Principal','Interest','Monthly EMI','Balance']]

    interest = loan_amount*(rate/1200)
    principal = emi_amount-interest
    balance = loan_amount-principal
    listed = [round(principal,2),round(interest,2),round(emi_amount,2),round(balance,2)]
    if balance > 0:
        main_list.append(listed)

    while balance>=0:
        interest = balance*(rate/1200)
        principal = emi_amount-interest
        balance = round(balance-principal,2)
        listed = [round(principal,2),round(interest,2),round(emi_amount,2),round(balance,2)]
        if balance > 0:
            main_list.append(listed)
    df = pd.DataFrame(main_list[1:],columns=main_list[:1])
    df['Month'] = range(1,df.shape[0]+1)
    year_range=[]
    for i in np.array(df.Month):
        year_range.append(int(math.ceil(i/12)))
    df['Year'] = year_range
    
    df = df[['Year','Month','Interest','Principal','Monthly EMI','Balance']]
    df.set_index([(       'Year',),(       'Month',)],inplace=True)
    

    return df


# max_loan_amount_on_capped_emi(emi_cap,roi,T = t)


with st.sidebar:
    st.title('EMI Calculator')
    menu = st.radio(
        "",
        ("Calculate EMI", "Calculate Max Loan Amount")
    )
if menu == "Calculate EMI":
    st.title('Calculate EMI')

    def update_slider_p():
        st.session_state.slider_p = st.session_state.numeric_p
    def update_numin_p():
        st.session_state.numeric_p = st.session_state.slider_p

    nlp,slp = st.columns([2,3])
    p = nlp.number_input("Rupees",100000,10000000,1000000,10000,key="numeric_p",on_change=update_slider_p)
    p = slp.slider("Loan Amount",100000,10000000,1000000,10000,key="slider_p",on_change=update_numin_p)



    def update_slider_r():
        st.session_state.slider_r = st.session_state.numeric_r
    def update_numin_r():
        st.session_state.numeric_r = st.session_state.slider_r

    nlr,slr = st.columns([2,3])

    r = nlr.number_input("%",0.1,30.0,6.5,0.05,key="numeric_r",on_change=update_slider_r)
    r = slr.slider("Rate of Interest",0.1,30.0,6.5,0.05,key="slider_r",on_change=update_numin_r)

    def update_slider_t():
        st.session_state.slider_t = st.session_state.numeric_t
    def update_numin_t():
        st.session_state.numeric_t = st.session_state.slider_t

    nlt,slt = st.columns([2,3])

    t = nlt.number_input("Years",1.0,30.0,5.0,0.25,key="numeric_t",on_change=update_slider_t)
    t = slt.slider("Time Period",1.0,30.0,5.0,0.25,key="slider_t",on_change=update_numin_t)


    x = emi(p,r,t)
    y = generate_monthly_installment_plan(p,r,t)
    y = y.reset_index()
    y = pd.DataFrame(y.values).rename(columns={0:"Year",1:"Month",2:"Interest",3:"Principal",4:"EMI Amount",5:"Balance"})
    emi_amt, loan_amt, tot_int = st.columns(3)
    emi_amt.metric("EMI amount",x)
    loan_amt.metric("Total Repayment Amount", round(x*t*12,2))
    tot_int.metric("Total Interest", round(y.iloc[:,2].sum()))
    piedf = pd.DataFrame({'Total Repayment Amount':['Loan Amount','Interest'],round(x*t*12,2):[p,round(y.iloc[:,2].sum())]})
    fig_pie = px.pie(piedf,names = 'Total Repayment Amount',values=round(x*t*12,2),labels=round(x*t*12,2))
    st.plotly_chart(fig_pie)

    pltdf = y[['Month','Interest','Principal']].melt(id_vars=['Month'],value_name='EMI Amount',value_vars=['Interest','Principal'],var_name='Interest & Principal')
    fig = px.bar(pltdf, x='Month', y='EMI Amount',color='Interest & Principal')
    st.plotly_chart(fig)


    installment_sheet = st.expander("Amortization plan")
    installment_sheet.table(data = y)
if menu == "Calculate Max Loan Amount":
    st.title("Calculate Max Loan Amount")
    def update_slider_ec():
        st.session_state.slider_ec = st.session_state.numeric_ec
    def update_numin_ec():
        st.session_state.numeric_ec = st.session_state.slider_ec

    nlec,slec = st.columns([2,3])
    ec = nlec.number_input("Rupees",1000,500000,10000,100,key="numeric_ec",on_change=update_slider_ec)
    ec = slec.slider("Maximum affordable EMI",1000,500000,10000,100,key="slider_ec",on_change=update_numin_ec)



    def update_slider_ur():
        st.session_state.slider_ur = st.session_state.numeric_ur
    def update_numin_ur():
        st.session_state.numeric_ur = st.session_state.slider_ur

    nlur,slur = st.columns([2,3])

    ur = nlur.number_input("%",0.1,30.0,6.5,0.05,key="numeric_ur",on_change=update_slider_ur)
    ur = slur.slider("Rate of Interest",0.1,30.0,6.5,0.05,key="slider_ur",on_change=update_numin_ur)

    def update_slider_te():
        st.session_state.slider_te = st.session_state.numeric_te
    def update_numin_te():
        st.session_state.numeric_te = st.session_state.slider_te

    nlte,slte = st.columns([2,3])

    te = nlte.number_input("Years",1.0,30.0,5.0,0.25,key="numeric_te",on_change=update_slider_te)
    te = slte.slider("Time Period",1.0,30.0,5.0,0.25,key="slider_te",on_change=update_numin_te)


    x1 = ec
    z = max_loan_amount_on_capped_emi(ec,ur,T = te)

    y1 = generate_monthly_installment_plan(z,ur,te)
    y1 = y1.reset_index()
    y1 = pd.DataFrame(y1.values).rename(columns={0:"Year",1:"Month",2:"Interest",3:"Principal",4:"EMI Amount",5:"Balance"})
    emi_amt, loan_amt, tot_int = st.columns(3)
    emi_amt.metric("Maximum Loan Amount",z)
    loan_amt.metric("Total Repayment Amount", round(x1*te*12,2))
    tot_int.metric("Total Interest", round(y1.iloc[:,2].sum()))

    piedf1 = pd.DataFrame({'Total Repayment Amount':['Loan Amount','Interest'],round(x1*te*12,2):[z,round(y1.iloc[:,2].sum())]})
    fig_pie1 = px.pie(piedf1,names = 'Total Repayment Amount',values=round(x1*te*12,2),labels=round(x1*te*12,2))
    st.plotly_chart(fig_pie1)

    pltdf1 = y1[['Month','Interest','Principal']].melt(id_vars=['Month'],value_name='EMI Amount',value_vars=['Interest','Principal'],var_name='Interest & Principal')
    fig1 = px.bar(pltdf1, x='Month', y='EMI Amount',color='Interest & Principal')
    st.plotly_chart(fig1)


    installment_sheet = st.expander("Amortization plan")
    installment_sheet.table(data = y1)





