# -*- coding: utf-8 -*-
import scrapy
import json
from scrapy.http import Request
from news_spider.items import NewsItem
from news_spider.start_urls import links
from news_spider.MyRedis import  MyRedis
import requests
import re
from lxml import etree
from scrapy.selector import Selector

class Spider(scrapy.Spider):
    name = 'spider'
    r=MyRedis()#初始化redis
    r.cache_redis()
    def __init__(self):
        self.start_urls = links


    def start_requests(self):
        for cate,url,source,tag in self.start_urls:
            yield scrapy.Request(url=url,meta={'url':url,'cate':cate,'source':source,'tag':tag},callback=self.parse)


    def parse(self, response):
        selector = Selector(response)
        url=response.meta['url']
        cate=response.meta['cate']
        source=response.meta['source']
        tag=response.meta['tag']
        atags=selector.xpath('//a/@href').extract()
        for a in atags:
            if not self.r.is_exit(a): #存在下一个链接
                continue
            if url in a:
                a=a[a.index('http'):]
                yield scrapy.Request(url=a, meta={'url': url, 'cate': cate, 'source': source, 'tag': tag},
                                     callback=self.parse_detail)
        return None

    def parse_detail(self,response):
        newsitem = NewsItem()
        selector = Selector(response)
        current_url=response.url#获取当前链接
        # print(current_url)
        url = response.meta['url']
        cate = response.meta['cate']
        source = response.meta['source']
        tag = response.meta['tag']
        tags = tag.split(':')
        xpath_rule = './/div[@' + tags[0] + '=' +"'"+tags[1]+"'"+']/p'
        res = selector.xpath(xpath_rule)
        content=''
        for c in res:
            content+=c.xpath('string(.)').extract_first()+'\r\n'
        content = re.sub('[\u3000 \xa0 \\t \u200b  ■]+', '', content)
        content = re.sub(r'showPlayer.*?;', '', content) #过滤人民网内特殊字符
        content = '\r\n'.join([c.replace('\n', '') for c in content.split('\r\n') if c.strip() and len(c.strip()) > 20])
        if content:
            newsitem['url'] = current_url
            newsitem['content'] = content
            newsitem['source'] = source
            newsitem['category'] = cate
            yield newsitem
        else:
            yield scrapy.Request(url=current_url, meta={'url': url, 'cate': cate, 'source': source, 'tag': tag},
                                 callback=self.parse)
