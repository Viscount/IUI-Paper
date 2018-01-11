# Datasets of Time-Sync Comment Videos in Bilibili.com

## Summary
These datasets describe Time-Sync Comments from Bilibili.com, a well-known TSC-enabled online video website in China. They contains 906 seasons, over 17,000 episodes and 32 millon comments. These data were created by over three millon users and collected between Aug. 23rd, 2017 and Sept. 3rd, 2017.</br>
The data are contained in the dump files of Mysql database, *bangumi.sql*, *episode.sql* and *danmaku.sql*. More details about the contents and use of all these files follows.</br>
Before using these data sets, please review their README files for the usage licenses and other details.
## Usage License
Neither the Tongji University nor any of the researchers involved can guarantee the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:

* The user may not state or imply any endorsement from the Tongji University.
* The user must acknowledge the use of the data set in publications resulting from the use of the data set (see below for citation information).
* The user may redistribute the data set, including transformations, so long as it is distributed under these same license conditions.
* The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from a faculty member of the GroupLens Research Project at the University of Minnesota.

In no event shall the Tongji University, its affiliates or employees be liable to you for any damages arising out of the use or inability to use these programs (including but not limited to loss of data or data being rendered inaccurate).

If you have any further questions or comments, please email lijf@tongji.edu.cn

## Citation
To acknowledge use of the dataset in publications, please cite the following paper (May be accessable after Mar. 11th, 2018):

* Liao J, Xian Y, Yang X, et al. TSCSet: A Crowdsourced Time-Sync Comment Dataset for Exploration of User Experience Improvement.</br>
DOI= http://dx.doi.org/10.1145/3172944.3172966

## Content and Use of Files

### Download

* BL-906:</br>
Baidu Clouddisk: https://pan.baidu.com/s/1jJujruA Access code: p9v7

### Formatting and Encoding
The dataset files are written as sql dump files with **utf8mb4** charset. It's recommended to use Mysql to import the data through following two ways:
* *mysqlimport* command:
> mysqlimport -u [uname] -p[pass] [dbname] [backupfile.sql]
* any database client software with import functions, such as *Mysql Workbench* and *Sequel Pro*.

### Bangumi (Season) Data Structure
* season_id: Primary key. The internal id of this season.
* cover: The url of cover image of this season.
* favorites: The number of user likes.
* is_finish: The status of this season, 1 for on air, 2 for finished.
* newest_ep_index: The index of the newest episode in this season.
* pub_time: The Unix timestamp of the season first released. 
* season_status: Status code of the season, the exact meaning still unknow.
* title: The name of this season.
* total_count: The number of episodes in this season.
* update_time: The timestamp of information update by Bilibili.
* url: The url of season detail page.
* week: The weekday that newest episode of this season is released.
* tags: Tags attached on this season, seperated by "|".
* actors: Voice actors in this season, seperated by "|".
* createdAt: The timestamp of current entry creation.
* updatedAt: The timestamp of current entry update.

### Episode Data Structure
* episode_id: Primary key. The internal id of this episode.
* av_id: The video id of this episode, used in matching url of episode page.
* cid: The comment id of this episode.
* coins: The number of user likes.
* cover: The url of cover image of this season.
* episode_status: The status of this episode, 1 for on air, 2 for finished.
* index: The index of this episode in season.
* index_title: The title of this episode.
* tags: Tags marked by users, seperated by "|".
* update_time: The timestamp of information update by Bilibili.
* season_id: The season_id of corresponding season.
* createdAt: The timestamp of current entry creation.
* updatedAt: The timestamp of current entry update.

### Danmaku (TSC) Data Structure
* raw_id: Primary key. The internal id of this TSC.
* playback_time: The time when this TSC appears in the video.
* type: The type of this TSC.
* font_size: The font size of this TSC.
* font_color: The color of this TSC, HTML color.
* unix_timestamp: The time of user published this TSC.
* pool: The position of how this TSC appear. 0 for normal, 1 for subtitle, 2 for advance.
* sender_id: The id of the sender who published this TSC.
* content: The content of TSC.
* episode_id: The episode_id of corresponding episode.
* createdAt: The timestamp of current entry creation.
* updatedAt: The timestamp of current entry update.
