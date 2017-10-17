# Importing data
# LOAD DATA LOCAL INFILE '/home/victor/na_data.csv' INTO TABLE na_data FIELDS TERMINATED BY ',' IGNORE 1 LINES;

import mysql.connector

# connect to database and create cursor object
cnx = mysql.connector.connect(user='root', password='',
                              host='localhost', database='msc_data')
cursor = cnx.cursor()

# query for unique values of gvkey
q_gvkey = """ SELECT gvkey FROM na_data GROUP BY gvkey """
cursor.execute(q_gvkey)

# list comprehension to append all unique values to list
keys = [gvkey[0] for gvkey in cursor]

# query for ebitda values grouped by gvkey
for i in keys:
    e_gvkey = " SELECT ebitda FROM na_data WHERE ebitda > '0' AND gvkey = (%s)" % keys[i]
    cursor.execute(e_gvkey, (keys,))
    ebitda = [float(ebitda[0]) for ebitda in cursor]
    print(len(ebitda))
    print(ebitda)


    # e_query = ("SELECT ebitda FROM na_data WHERE ebitda > '0' AND gvkey = 'key'")
    # cursor.execute(e_query)

