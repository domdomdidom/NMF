import pandas as pd
import numpy as np
from sklearn.decomposition import NMF as NMF_sklearn
import warnings; warnings.simplefilter('ignore')
from collections import Counter

class ReadMyFiles():
    ''' These will format BigCommerce output files specifically :) '''

    def __init__(self):

        self.custy_df = None
        self.order_df = None
        self.subscriber_df = None
        self.product_df = None

    def read_customer(self, filepath):

        custy_df = pd.read_csv(filepath, skiprows=None, index_col='Customer ID', parse_dates=["Date Joined"]).dropna(axis=0, how='all')
        self.custy_df = custy_df.drop(columns=['Rep Name', 'Education'], axis=1)
        return self.custy_df
    
    def read_order(self, filepath):

        self.order_df = pd.read_csv(filepath, parse_dates=["Order Date"])
        return self.order_df

    def read_product(self, filepath):

        self.product_df = pd.read_csv(filepath)
        return self.product_df

    def read_marketing(self, filepath):

        self.subscriber_df= pd.read_csv(filepath)
        return self.subscriber_df

def make_historical_purchase_matrix(custy_df, order_df, product_df):
    
    historical_purchase_df = pd.DataFrame(0, index=custy_df.index, columns=product_df['Product ID'])

    for customer in custy_df.index.values:

        mask = order_df[order_df['Customer ID'] == customer] # mask for all orders under a customer

        for order in mask['Product Details'].values:
            itemized = order.split('|') # split each "itemized order line"

            for line in itemized:
                keep, rubbish = line.split(', Product SKU') # get rid of everything after prodct SKU

                prod_id, prod_qty = keep.split(',')

                rubbish, prod_id = prod_id.split(':') # 
                rubbish, prod_qty  = prod_qty.split(':')

                prod_id = int(prod_id.strip()) # strip whitespace
                prod_qty = int(prod_qty.strip())

                if prod_id not in list(product_df['Product ID']):
                    pass
                
                else: historical_purchase_df[prod_id][customer] += prod_qty
                    
    historical_purchase_matrix = historical_purchase_df.as_matrix()                   
    print("historical itemized matrix assembled.") 

    return historical_purchase_df, historical_purchase_matrix

def do_NMF(historical_purchase_matrix, product_df, get_top_products=True):
        
    nmf = NMF_sklearn(n_components=5, max_iter=450)
    W = nmf.fit_transform(historical_purchase_matrix) # how much each customer belongs to each "topic"
    H = nmf.components_ # how much each item belongs to each "topic"

    if get_top_products == True:
        
        print("Here are the top products for %s topics" % (5))
        for topic in range(0, 5):
            indicies = H[topic].argsort()[-25:]
            print("\n")
            print(product_df['Name'][indicies])

    return W, H

def get_items_associated(historical_purchase_matrix, feature_df, product_df, n_topics=5, max_iters=350, n_churniest_topics=3, n_churniest_items=25):
    ''' 
    Get products most associated with a particular group
    Parameters
    ----------
    index of a dataframe slice (pandas.core.indexes.numeric.Int64Index)

        
    Attributes
    ----------  
    returns "most associated" items. I often use this for churn.

    '''
    slice = feature_df.index.astype(int)

    nmf = NMF_sklearn(n_components=n_topics, max_iter=max_iters, alpha=0.0)
    W = nmf.fit_transform(historical_purchase_matrix) # how much each customer belongs to each "topic"
    H = nmf.components_ # how much each item belongs to each "topic"
    
    sums = W[slice].sum(axis=0)
    churniest_topics = sums.argsort()[-n_churniest_topics:] 
    
    c = Counter()
    
    for topic in churniest_topics:
        indicies = H[topic].argsort()[-50:]
    
        for product in product_df['Name'][indicies]:
            c[product] += 1
            
    return c.most_common(n_churniest_items)
