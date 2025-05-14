pip install pandas matplotlib statsmodels openai fredapi streamlit python-dotenv
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import openai
from fredapi import Fred
import streamlit as st

# 直接使用你的API密钥

FRED_API_KEY = "31c7fd971415f73bf256e909c6418669"

openai.api_key = OPENAI_API_KEY
fred = Fred(api_key=FRED_API_KEY)

# Streamlit界面
st.title("📈 AI驱动的经济趋势分析助手")

indicator = st.text_input("请输入经济数据代码（如CPIAUCSL, GDP, UNRATE）：", "CPIAUCSL")
start_date = st.text_input("起始日期（如2010-01-01）：", "2010-01-01")

if st.button("获取并分析数据"):
    data = fred.get_series(indicator, observation_start=start_date)
    df = pd.DataFrame(data, columns=[indicator])
    df.dropna(inplace=True)

    st.write(f"✅ 已成功获取 {indicator} 数据：")
    st.line_chart(df)

    df['time'] = range(len(df))
    X = sm.add_constant(df['time'])
    y = df[indicator]

    model = sm.OLS(y, X).fit()

    st.subheader("📊 回归分析结果")
    st.write(model.summary().tables[1])

    df['predicted'] = model.predict(X)
    plt.figure(figsize=(10, 5))
    plt.plot(df[indicator], label='Actual')
    plt.plot(df['predicted'], label='Predicted', linestyle='--')
    plt.legend()
    plt.title(f"{indicator}趋势与预测")
    st.pyplot(plt)

    st.subheader("🤖 AI生成的经济分析报告")
    prompt = f"""
    你是一名资深经济分析师，下面是一组经济数据的回归分析结果：

    {model.summary()}

    请你用简单、易懂的语言解释：
    - 数据呈现的趋势是什么？
    - 预测结果有何经济意义？
    - 你对政策制定者有什么建议？
    """

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    analysis = completion.choices[0].message.content
    st.write(analysis)

st.subheader("💬 互动问答小助手")
user_question = st.text_input("你可以向AI提出关于经济数据或政策的问题：", 
                              "为什么利率升高可能降低通货膨胀？")

if st.button("询问AI"):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一名经济学教授，用简单易懂的语言回答学生的问题。"},
            {"role": "user", "content": user_question}
        ]
    )
    answer = completion.choices[0].message.content
    st.write("🔍 AI的回答：")
    st.write(answer)
