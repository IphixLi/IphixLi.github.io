---
title: Korean drama Normalization
date: 2022-08-16 13:06:24
tags:
- Data normalization
- sql
- databases
- data engineering
- k-drama
- Pandas
categories:
- Projects
toc: true
cover: /gallery/liberation_notes.jpg
thumbnail: /gallery/liberation_notes.jpg
---
<p class="not-gallery-item" style="font-size:13px" align="center"><em>(on the cover) My Liberation Notes</em></p>

<em>TLDR: I normalized korean drama list dataset for easy querying and accessibility. I created individual genres,tags,platforms,aired_on, and imdb tables to eliminate arrays in table.</em>

<!-- more -->

In RDMBS design, usability plays a big role in data modeling. You have to consider the data stakeholders involved and their use case such as how frequently queries will be performed, new data instances will be inserted, updated or deleted. Data normalization serves this purpose.


Nested elements and list elements in RDBMS are notoriously difficult to query. Most of the time, there is not-so-pretty destructuring and joins along the way just to make a simple query on elements with compound data types such as arrays which ends up making querying and updating compound entries take a lot of computing time and resources.


Let’s take a look at a part of kdrama list.
<em>full kdramalist on [kaggle](https://www.kaggle.com/datasets/iphigeniebera/korean-drama-list-about-740-unique-dramas)</em>
<p align="center" class="mb-2">
<img class="not-gallery-item" style="width:100%;height:80%" src="https://iphixli.github.io/gallery/kdrama_part.png">
</p>

Also parsing different file types into sql statements becomes a challenge for compound datatypes such as lists in csv. Csv by default  stores elements as strings, so lists become strings, and straight forward queries such as indexing access are a challenge if not even impossible without string manipulation preprocessing.

Also for convenience to people who accessesses your data, it is better to keep a table at atomic level for each cell and provide structure for altering and accessing data without necessarily understanding the whole structure of data artifact or tampering data intergrity, this is where data normalization comes into play

According to [wikipedia](https://en.wikipedia.org/wiki/Database_normalization),

“Database normalization is the process of restructuring a relational database in accordance with a series of so-called normal forms in order to reduce data redundancy and improve data integrity. It was first proposed by Edgar F. Codd as an integral part of his relational model. This leads to the creation of duplicate values and multiple tables.

There are many types of normalization but the most commonly used are 3.

<strong>First normal form (1NF)</strong>: requires that a table satisfies the following conditions: 
- There is duplicated data: no two rows have same values for each attribute.
- All columns are “regular” with no hidden values: this means that each cell contains a single value.
1NF normalization created atomic values however can lead to creation of lots of rows as you need to account for possible combinations in case of multiple values destructuring.

For example, 1NF of kdramalist table would look like this:
<p align="center" class="mb-2">
<img class="not-gallery-item" style="width:100%;height:80%" src="https://iphixli.github.io/gallery/firstNdramas.png">
</p>

Notice how permutations between tags, aired on and actors attributes. While this gives us a single element for each cell, we end up getting a large number of rows. For the kdrama list which had about 743 rows, 1NF led to ~510,000 rows, about 650  times the original list. Imagine if we had an extensive number of tags, genres for each kdrama, we can jump to gigabyte file in instant, oof!!!!.  This is not to mention if you need to update an attribute in this kind of table, copying over elements and checking for duplicates would be a nightmare. This is where the second normal form comes for a save.

<strong>Second normal form (2NF)</strong>: An entity is in a second normal form if all of its attributes depend on the whole primary key. So this means that the values in the different columns have a dependency on the other columns.
- The table must be already in 1 NF and all non-key columns of the tables must depend on the PRIMARY KEY.
- The partial dependencies are removed and placed in a separate table.

2NF strictly guides us to keep only columns which depend on primary keys and keep others in separate tables. This is why we have created additional tables: actors, genres, tags, platforms, aired_on, imdb and one table main_descriptors containing attributes that depend on the primary key, imdb_name. To enforce primary keys on  actors, genres, tags, platforms, aired_on, and imdb tables, one can create composite keys using sql.

<strong>Third normal form (3NF)</strong>: you should eliminate fields in a table that do not depend on the key.
- A Table is already in 2 NF
- Non-Primary key columns shouldn’t depend on the other non-Primary key columns
- There is no transitive functional dependency.
If A->B and B->C are the two functional dependencies, then A->C is called the Transitive Dependency. Our current tables satisfy the above conditions therefore all of them are in 3rd normal form.

Main_descriptors actors, genres,tags,platforms,aired_on, and imdb tables can be found in [Kaggle](https://www.kaggle.com/datasets/iphigeniebera/korean-drama-list-about-740-unique-dramas). and this [github repo](https://github.com/IphixLi/kdrama_project) under normalized_tables folder and generating code in normalizing.py.
 
 you can read on  how korean drama dataset was generated by webscraping in this [post](https://iphixli.github.io/Projects/korean-drama-webscraping/).

 ********************************************************************************************************************
<em>This data normalization project is part of Kdrama project in which I practice my data science and engineering skills, step by step until I achieve highly accurate model for recommending korean drama.</em>

you can watch projects associated with k-drama projects by following [k-drama tag](https://iphixli.github.io/tags/k-drama/)


