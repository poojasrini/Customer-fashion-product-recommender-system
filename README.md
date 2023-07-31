# Recommendation-system-for-fashion-products
Applied machine learning group project on H&amp;M dataset

## Data
https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data

articles.csv: table

transactions.csv: images

customers.csv: images

Transactions.csv summary:

- Rows: each row represents a transaction of the purchases in H&M group.
- Columns:
  - t_dat: date of the transaction
  - customer_id: a foreign key that maps to each customer entry in the customer_id column in customers.csv, which contains customers' metadata
  - article_id: a foreign key that maps to each customer entry in the article_id column in articles.csv, which contains articles' metadata
  - price: price that the customer paid for the transaction, there are multiple prices for the same article, even on the same day/channel. According to data's contributor, the unit of price is not any "currency/unit" since they chose to not disclose the real values
  - sales_channel_id: it determines how the customer purchases the article, 2 is online and 1 store

**<span style="color:#023e8a;"> The article.csv contains all h&m articles with details such as a type of product, a color, a product group and other features.</span>**  
**<span style="color:#023e8a;"> Article data description: </span>**

> `article_id` **<span style="color:#023e8a;">: A unique identifier of every article.</span>**  
> `product_code`, `prod_name` **<span style="color:#023e8a;">: A unique identifier of every product and its name (not the same).</span>**  
> `product_type`, `product_type_name` **<span style="color:#023e8a;">: The group of product_code and its name</span>**  
> `graphical_appearance_no`, `graphical_appearance_name` **<span style="color:#023e8a;">: The group of graphics and its name</span>**  
> `colour_group_code`, `colour_group_name` **<span style="color:#023e8a;">: The group of color and its name</span>**  
> `perceived_colour_value_id`, `perceived_colour_value_name`, `perceived_colour_master_id`, `perceived_colour_master_name` **<span style="color:#023e8a;">: The added color info</span>**  
> `department_no`, `department_name`: **<span style="color:#023e8a;">: A unique identifier of every dep and its name</span>**  
> `index_code`, `index_name`: **<span style="color:#023e8a;">: A unique identifier of every index and its name</span>**  
> `index_group_no`, `index_group_name`: **<span style="color:#023e8a;">: A group of indeces and its name</span>**  
> `section_no`, `section_name`: **<span style="color:#023e8a;">: A unique identifier of every section and its name</span>**  
> `garment_group_no`, `garment_group_name`: **<span style="color:#023e8a;">: A unique identifier of every garment and its name</span>**  
> `detail_desc`: **<span style="color:#023e8a;">: Details</span>**  

