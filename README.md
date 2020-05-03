# Disaster Response Pipeline Project

### Project Description:
项目分析了来自 Figure 8 的真实灾害消息数据集，构建了一个对灾害消息进行分类的模型，并用于网络应用程序。
项目构建了一个网站，使用可视化，展示了灾害消息数据的基本情况。
同时，灾害应对工作人员可以使用该网站，查询一条新的灾害消息所属的灾害分类结果，以便同相关机构对消息做出响应。

### File Description
1. 数据文件
    - data/disaster_categories.csv
    - data/disaster_messages.csv
    - DisasterResponse.db:上述两个csv合并清洗后，写入的数据库，库中表名DisasterResponse

2. py文件
    - data/process_data.py: ETL模块，清洗数据
    - models/train_classifier.py: ML模块，训练和输出模型
    - app/run.py: 创建了2个数据图，渲染前端页面

3. html文件
    - app/templates/master.html: 前端默认主页
    - app/templates/go.html: 前端消息查询结果页， 示例查询：Help! A child fell  into the water! 

4. notbook文件
    - models/ML Pipeline Preparation-zh.ipynb: ETL模块过程代码
    - data/ETL Pipeline Preparation-zh.ipynb: ML模块过程代码

### Instructions:
1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Enter into the "app" directory. Run the following command to run web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Web page Screenshots:
![Image text](https://raw.githubusercontent.com/lynnliuuu/UdacityDSND/master/app/home_graphs_screenshot.png)
![Image text](https://raw.githubusercontent.com/lynnliuuu/UdacityDSND/master/app/query_screenshot.png)

