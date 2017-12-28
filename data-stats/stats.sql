-- Add index
create index idx_sender on danmaku(sender_id);

-- Count number of tsc per minite in episodes
create table episode_danmaku_cnt_dd
as 
 select episode_id, ceil(playback_time/60) as t_min
    from danmaku ;
create table episode_danmaku_cnt_summary
as 
 select episode_id, t_min, count(1) as danmaku_cnt
    from episode_danmaku_cnt_dd
group by episode_id, t_min ;

-- count average number of episodes each user watched
create table sender_season_stats 
as 
select sender_id, count(1) as season_cnt 
from 
(
    select sender_id, season_id
    from 
    (
        select season_id, episode_id
        from episode
    )a
    join 
    (
        select sender_id, episode_id
        from danmaku 
    )b 
    on a.episode_id = b.episode_id
    group by sender_id, season_id
)t
group by sender_id;

-- Count number of tscs every user published
create table sender_danmaku_stats 
as 
select sender_id, count(*) as danmaku_cnt
    from danmaku 
    group by sender_id;
