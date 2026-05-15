import csv
import json
import logging
import shutil
import time
from pathlib import Path

import requests
from tqdm import tqdm
from newspaper import Article

from util.util import DataCollector
from util.util import Config, create_dir
from util import Constants


def crawl_link_article(url):
    result_json = None

    try:
        if 'http' not in url:
            if url[0] == '/':
                url = url[1:]
            try:
                article = Article('http://' + url)
                article.download()
                time.sleep(2)
                article.parse()
                flag = True
            except:
                logging.exception("Exception in getting data from url {}".format(url))
                flag = False
                pass
            if flag == False:
                try:
                    article = Article('https://' + url)
                    article.download()
                    time.sleep(2)
                    article.parse()
                    flag = True
                except:
                    logging.exception("Exception in getting data from url {}".format(url))
                    flag = False
                    pass
            if flag == False:
                return None
        else:
            try:
                article = Article(url)
                article.download()
                time.sleep(2)
                article.parse()
            except:
                logging.exception("Exception in getting data from url {}".format(url))
                return None

        if not article.is_parsed:
            return None

        visible_text = article.text
        top_image = article.top_image
        images = article.images
        keywords = article.keywords
        authors = article.authors
        canonical_link = article.canonical_link
        title = article.title
        meta_data = article.meta_data
        movies = article.movies
        publish_date = article.publish_date
        source = article.source_url
        summary = article.summary

        result_json = {'url': url, 'text': visible_text, 'images': list(images), 'top_img': top_image,
                       'keywords': keywords,
                       'authors': authors, 'canonical_link': canonical_link, 'title': title, 'meta_data': meta_data,
                       'movies': movies, 'publish_date': get_epoch_time(publish_date), 'source': source,
                       'summary': summary}
    except:
        logging.exception("Exception in fetching article form URL : {}".format(url))

    return result_json


def get_epoch_time(time_obj):
    if time_obj:
        return time_obj.timestamp()

    return None


def get_web_archieve_results(search_url):
    try:
        archieve_url = "http://web.archive.org/cdx/search/cdx?url={}&output=json".format(search_url)

        response = requests.get(archieve_url)
        response_json = json.loads(response.content)

        response_json = response_json[1:]

        return response_json

    except:
        return None


def get_website_url_from_arhieve(url):
    """ Get the url from http://web.archive.org/ for the passed url if exists."""
    archieve_results = get_web_archieve_results(url)
    if archieve_results:
        modified_url = "https://web.archive.org/web/{}/{}".format(archieve_results[0][1], archieve_results[0][2])
        return modified_url
    else:
        return None


def crawl_news_article(url):
    news_article = crawl_link_article(url)

    # If the news article could not be fetched from original website, fetch from archive if it exists.
    if news_article is None:
        archieve_url = get_website_url_from_arhieve(url)
        if archieve_url is not None:
            news_article = crawl_link_article(archieve_url)

    return news_article


def append_news_content_failure(log_path, news_source, label, news, folder_path, reason):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    with log_path.open("a", encoding="utf-8", newline="") as f:
        fieldnames = ["news_source", "label", "news_id", "title", "url", "folder_path", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "news_source": news_source,
            "label": label,
            "news_id": news.news_id,
            "title": news.news_title,
            "url": news.news_url,
            "folder_path": str(folder_path),
            "reason": reason,
        })


def remove_failed_news_folder(news_dir):
    news_dir = Path(news_dir)
    if news_dir.exists() and news_dir.is_dir():
        shutil.rmtree(str(news_dir))


def collect_news_articles(news_list, news_source, label, config: Config):
    create_dir(config.dump_location)
    create_dir("{}/{}".format(config.dump_location, news_source))
    create_dir("{}/{}/{}".format(config.dump_location, news_source, label))

    save_dir = Path("{}/{}/{}".format(config.dump_location, news_source, label))
    failure_log_path = Path(config.dump_location) / "logs" / "news_content_failures.csv"

    for news in tqdm(news_list):
        news_dir = save_dir / news.news_id
        create_dir(str(news_dir))

        news_article = crawl_news_article(news.news_url)
        if news_article:
            json.dump(
                news_article,
                open(str(news_dir / "news content.json"), "w", encoding="UTF-8"),
            )
        else:
            append_news_content_failure(
                failure_log_path,
                news_source,
                label,
                news,
                news_dir,
                reason="crawl_or_parse_failed",
            )
            logging.info(
                "Failed to collect news content. Removed folder. source=%s label=%s news_id=%s url=%s",
                news_source,
                label,
                news.news_id,
                news.news_url,
            )
            remove_failed_news_folder(news_dir)


class NewsContentCollector(DataCollector):

    def __init__(self, config):
        super(NewsContentCollector, self).__init__(config)

    def collect_data(self, choices):
        for choice in choices:
            news_list = self.load_news_file(choice)
            collect_news_articles(news_list, choice["news_source"], choice["label"], self.config)
