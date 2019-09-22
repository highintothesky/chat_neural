# scraping the internetCommentEtiq subreddit
import praw

reddit = praw.Reddit('DEFAULT', user_agent='scraper bot user agent')
out_f = open('data/subreddit_comments.txt', 'w')

hot_posts = reddit.subreddit('InternetCommentEtiq').top(limit=1000)
for post in hot_posts:
    for comm in post.comments:
        out_f.write(comm.body)
        out_f.write('\n')
        # print(comm.body)
    # print(post.comments)
