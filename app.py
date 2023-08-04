import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import pandas as pd
import numpy as np

st.sidebar.title("Whatsapp Chat And Sentiment Analyzer")

# VADER : is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments.
nltk.download('vader_lexicon')

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

     # Importing SentimentIntensityAnalyzer class from "nltk.sentiment.vader"
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    # Object
    sentiments = SentimentIntensityAnalyzer()
    
    # Creating different columns for (Positive/Negative/Neutral)
    data["po"] = [sentiments.polarity_scores(i)["pos"] for i in data["message"]] # Positive
    data["ne"] = [sentiments.polarity_scores(i)["neg"] for i in data["message"]] # Negative
    data["nu"] = [sentiments.polarity_scores(i)["neu"] for i in data["message"]] # Neutral
    
    # To indentify true sentiment per row in message column
    def sentiment(d):
        if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
            return 1
        if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
            return -1
        if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
            return 0

    # Creating new column & Applying function
    data['value'] = data.apply(lambda row: sentiment(row), axis=1)
    
    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")

        col1,col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
            st.pyplot(fig)
            
            
        #Sentiment Analysis
        # Monthly activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Activity map(Positive)</h3>",unsafe_allow_html=True)
            
            busy_month = helper.month_activity_map2(selected_user, data,1)
            
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Activity map(Neutral)</h3>",unsafe_allow_html=True)
            
            busy_month = helper.month_activity_map2(selected_user, data, 0)
            
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Activity map(Negative)</h3>",unsafe_allow_html=True)
            
            busy_month = helper.month_activity_map2(selected_user, data, -1)
            
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Daily activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: white;'>Daily Activity map(Positive)</h3>",unsafe_allow_html=True)
            
            busy_day = helper.week_activity_map2(selected_user, data,1)
            
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: white;'>Daily Activity map(Neutral)</h3>",unsafe_allow_html=True)
            
            busy_day = helper.week_activity_map2(selected_user, data, 0)
            
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: white;'>Daily Activity map(Negative)</h3>",unsafe_allow_html=True)
            
            busy_day = helper.week_activity_map2(selected_user, data, -1)
            
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Weekly activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                st.markdown("<h3 style='text-align: center; color: white;'>Weekly Activity Map(Positive)</h3>",unsafe_allow_html=True)
                
                user_heatmap = helper.activity_heatmap2(selected_user, data, 1)
                
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.text('error.webp')
        with col2:
            try:
                st.markdown("<h3 style='text-align: center; color: white;'>Weekly Activity Map(Neutral)</h3>",unsafe_allow_html=True)
                
                user_heatmap = helper.activity_heatmap2(selected_user, data, 0)
                
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.text('error.webp')
        with col3:
            try:
                st.markdown("<h3 style='text-align: center; color: white;'>Weekly Activity Map(Negative)</h3>",unsafe_allow_html=True)
                
                user_heatmap = helper.activity_heatmap2(selected_user, data, -1)
                
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.text('error.webp')

        # Daily timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: white;'>Daily Timeline(Positive)</h3>",unsafe_allow_html=True)
            
            daily_timeline = helper.daily_timeline2(selected_user, data, 1)
            
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: white;'>Daily Timeline(Neutral)</h3>",unsafe_allow_html=True)
            
            daily_timeline = helper.daily_timeline2(selected_user, data, 0)
            
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: white;'>Daily Timeline(Negative)</h3>",unsafe_allow_html=True)
            
            daily_timeline = helper.daily_timeline2(selected_user, data, -1)
            
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Monthly timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Timeline(Positive)</h3>",unsafe_allow_html=True)
            
            timeline = helper.monthly_timeline2(selected_user, data,1)
            
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Timeline(Neutral)</h3>",unsafe_allow_html=True)
            
            timeline = helper.monthly_timeline2(selected_user, data,0)
            
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: white;'>Monthly Timeline(Negative)</h3>",unsafe_allow_html=True)
            
            timeline = helper.monthly_timeline2(selected_user, data,-1)
            
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Percentage contributed
        if selected_user == 'Overall':
            col1,col2,col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: white;'>Most Positive Contribution</h3>",unsafe_allow_html=True)
                x = helper.percentage2(data, 1)
                
                # Displaying
                st.dataframe(x)
            with col2:
                st.markdown("<h3 style='text-align: center; color: white;'>Most Neutral Contribution</h3>",unsafe_allow_html=True)
                y = helper.percentage2(data, 0)
                
                # Displaying
                st.dataframe(y)
            with col3:
                st.markdown("<h3 style='text-align: center; color: white;'>Most Negative Contribution</h3>",unsafe_allow_html=True)
                z = helper.percentage2(data, -1)
                
                # Displaying
                st.dataframe(z)


        # Most Positive,Negative,Neutral User...
        if selected_user == 'Overall':
            
            # Getting names per sentiment
            x = data['user'][data['value'] == 1].value_counts().head(10)
            y = data['user'][data['value'] == -1].value_counts().head(10)
            z = data['user'][data['value'] == 0].value_counts().head(10)

            col1,col2,col3 = st.columns(3)
            with col1:
                # heading
                st.markdown("<h3 style='text-align: center; color: white;'>Most Positive Users</h3>",unsafe_allow_html=True)
                
                # Displaying
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                # heading
                st.markdown("<h3 style='text-align: center; color: white;'>Most Neutral Users</h3>",unsafe_allow_html=True)
                
                # Displaying
                fig, ax = plt.subplots()
                ax.bar(z.index, z.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                # heading
                st.markdown("<h3 style='text-align: center; color: white;'>Most Negative Users</h3>",unsafe_allow_html=True)
                
                # Displaying
                fig, ax = plt.subplots()
                ax.bar(y.index, y.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

        # WORDCLOUD......
        col1,col2,col3 = st.columns(3)
        with col1:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: white;'>Positive WordCloud</h3>",unsafe_allow_html=True)
                
                # Creating wordcloud of positive words
                df_wc = helper.create_wordcloud2(selected_user, data,1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')
        with col2:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: white;'>Neutral WordCloud</h3>",unsafe_allow_html=True)
                
                # Creating wordcloud of neutral words
                df_wc = helper.create_wordcloud2(selected_user, data,0)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')
        with col3:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: white;'>Negative WordCloud</h3>",unsafe_allow_html=True)
                
                # Creating wordcloud of negative words
                df_wc = helper.create_wordcloud2(selected_user, data,-1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')

        # Most common positive words
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                # Data frame of most common positive words.
                most_common_df = helper.most_common_words2(selected_user, data,1)
                
                # heading
                st.markdown("<h3 style='text-align: center; color: white;'>Positive Words</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1],color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')
        with col2:
            try:
                # Data frame of most common neutral words.
                most_common_df = helper.most_common_words2(selected_user, data,0)
                
                # heading
                st.markdown("<h3 style='text-align: center; color: white;'>Neutral Words</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1],color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')
        with col3:
            try:
                # Data frame of most common negative words.
                most_common_df = helper.most_common_words2(selected_user, data,-1)
                
                # heading
                st.markdown("<h3 style='text-align: center; color: white;'>Negative Words</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.text('error.webp')











