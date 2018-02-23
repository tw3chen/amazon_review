import scrapy


class ConsumerAffairsAmazonReviewSpider(scrapy.Spider):
    name = "conumeraffairsamazonreviewspider"
    start_urls = ["https://www.consumeraffairs.com/online/amazon.html"]

    # def parse(self, response):
    #     for div in response.css("div.rvw-bd"):
    #         raw_review_text_list = div.css("p ::text").extract()
    #         raw_review_text = ' '.join(raw_review_text_list)
    #         review_text = raw_review_text.replace("\n", " ").replace("\t", "")
    #         yield {'review_text': review_text}
    #     next_page_partial_links = response.css('nav.prf-pgr > a[rel=next]::attr(href)').extract()
    #     if next_page_partial_links:
    #         partial_link = next_page_partial_links[0]
    #         link = "https://www.consumeraffairs.com" + partial_link
    #         yield response.follow(link, self.parse)

    def parse(self, response):
        review_texts = []
        for div in response.css("div.rvw-bd"):
            raw_review_text_list = div.css("p ::text").extract()
            raw_review_text = ' '.join(raw_review_text_list)
            review_text = raw_review_text.replace("\n", " ").replace("\t", "")
            review_texts.append(review_text)
        review_ratings = []
        for raw_review_rating in response.css("div[data-rating]::attr(data-rating)"):
            review_rating = raw_review_rating.extract()
            if review_rating.endswith(".0"):
                review_ratings.append(int(float(review_rating)))
                print(review_rating)
        for review_text, review_rating in zip(review_texts, review_ratings):
            yield {'review_text': review_text, 'review_rating': review_rating}
        next_page_partial_links = response.css('nav.prf-pgr > a[rel=next]::attr(href)').extract()
        if next_page_partial_links:
            partial_link = next_page_partial_links[0]
            link = "https://www.consumeraffairs.com" + partial_link
            yield response.follow(link, self.parse)


# scrapy runspider -o consumer_affairs_amazon_review.csv -t csv consumer_affairs_amazon_review_spider.py
