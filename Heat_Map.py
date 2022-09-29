import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

def distance(date1, date2, df):
	'''
	Calculates the Euclidian distance between a point corresponding to selected_date and all other points in the input df
	Input: Input dataframe df with transformed indicators (rows: dates; columns: indicators)
	Output: Returns the Euclidian distance
	'''
	
	#Get data at selected_date
	df_date1=df.loc[date1]
	df_date2=df.loc[date2]
	
	#Calculate Euclidian norm	
	return np.linalg.norm(df_date1.values-df_date2.values)

def transform_data(LT_history=60):
	'''
	Input: Reads an input spreadsheet with raw data to transform
		   For each column, one calculates the level vs long term history and the rate of change vs the trailing year
		   Data is then zscored and later on in the main code weighted using the input weights vector
	Output: Returns the df of all transformed columns (rows: dates; columns: indicators x 2 (one for level, one for rate of change))
	'''
	
	#Import all monthly data
	df = pd.read_excel("Heat_Map_data_new.xlsx",sheet_name="Data",index_col=0,header=0,skiprows=1,usecols="B:N")
	df.index=pd.to_datetime(df.index)
	df.index=df.index.map(lambda t: t.replace(day=1))
	
	#Import quarterly data (Real GDP) and convert it to monthly
	df_q = pd.read_excel("Heat_Map_data_new.xlsx",sheet_name="Data",index_col=0,header=0,skiprows=1,usecols="R,S")
	df_q.index=pd.to_datetime(df_q.index)
	df_q=df_q.dropna()
	df_q=df_q.resample('M').pad()
	df_q.index=df_q.index.map(lambda t: t.replace(day=1))
	df_q.columns=['Real GDP']
	df=pd.concat([df, df_q],axis=1,join='outer')
	
	#Calculate 10Y-2Y
	df['10Y2Y']=df['USGG10YR Index']-df['USGG2YR Index']
	
	#Invert PE
	df['Fwd_PE']=1/df['Fwd_PE']
	df['Trailing_PE']=1/df['Trailing_PE']
	
	#Take log of VIX, VXO and OAS
	for col in ['VIX Index', 'VXO Index', 'LUACOAS Index', 'LF98OAS Index']:
		df[col]=np.log(df[col])
	
	#Format all as float
	df=df.astype(float)
	
	#Calculate level vs long term history
	df_level = df - df.rolling(window=LT_history, min_periods=1).mean()
	
	#Calculate rate of change vs the trailing year
	df_ror = df.pct_change(periods = 12) #for every month t, calculates (x_t-x_(t-nb_periods))/x_t	
	df_ror.columns = [str(col) + ' ror' for col in df_ror.columns]
	
	#Concatenate both DataFrames
	df=pd.concat([df_level, df_ror],axis=1,join='outer')
		
	#z-score all measures
	df = (df-df.mean(axis = 0, skipna=True))/df.std(axis = 0, skipna=True)
	df.fillna(0., inplace=True)

	return df

def define_weights_categories():
	
	weights={
	#weights for level
	'VIX Index': 0.125, 
	'VXO Index': 0., 
	'Fwd_PE': 0.0625, 
	'Trailing_PE': 0.0625, 
	'LUACOAS Index': 0.0625,
    'LF98OAS Index': 0.0625, 
    'USGG2YR Index': 0.0625, 
    'USGG10YR Index': 0.,
    '10Y2Y': 0.0625,
    'Real GDP': 0.0833,
    'CPUPXCHG Index': 0.0833,
    'USURTOT Index': 0.0833, 
    'OEUSLCAC Index': 0.125, 
    'OEUSDHAO Index': 0.125,
 	#weights for ror   
    'VIX Index ror': 0.125, 
	'VXO Index ror': 0., 
	'Fwd_PE ror': 0.0625, 
	'Trailing_PE ror': 0.0625, 
	'LUACOAS Index ror': 0.0625,
    'LF98OAS Index ror': 0.0625, 
    'USGG2YR Index ror': 0.0625, 
    'USGG10YR Index ror': 0.,
    '10Y2Y ror': 0.0625,
    'Real GDP ror': 0.0833,
    'CPUPXCHG Index ror': 0.0833,
    'USURTOT Index ror': 0.0833, 
    'OEUSLCAC Index ror': 0.125, 
    'OEUSDHAO Index ror': 0.125
	}
	
	market= [
	'VIX Index',
	'Fwd_PE', 
	'Trailing_PE', 
	'LUACOAS Index',
    'LF98OAS Index', 
    'USGG2YR Index', 
    '10Y2Y'
	]
	
	economic= [
	'Real GDP',
	'CPUPXCHG Index ror',
    'USURTOT Index ror' 
	]
	
	confidence = [
	'OEUSLCAC Index',
    'OEUSDHAO Index'
	]

	return weights, market, economic, confidence

def calc_distances(df_total, market, economic, confidence, reference_date='2018-04-01'):
    #Sub-categories:
	df_market=df_total[[col_name for col_name in market]]
	df_economic=df_total[[col_name for col_name in economic]]
	df_confidence=df_total[[col_name for col_name in confidence]]
	
	#Create df dist (rows: dates; columns: distances from a specific date, default '2018-04-01')
	distances=pd.DataFrame(index=df_total.index)
	
	for date in df_total.index:
		distances.loc[date, "Market"]=distance(reference_date, date, df_market)
		distances.loc[date, "Economic"]=distance(reference_date, date, df_economic)
		distances.loc[date, "Confidence"]=distance(reference_date, date, df_confidence)		
		distances.loc[date, "Total"]=distance(reference_date, date, df_total)
	
	distances.index=pd.to_datetime(distances.index)
	
	return distances

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	From http://chris35wills.github.io/matplotlib_diverging_colorbar/
	Normalise the colorbar so that diverging bars work their way either side from a prescribed midpoint value
    leading to a better harmonised color map
	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def plot_graphs(distances):
	#Initialise plot
	fig, [ax_market, ax_economic, ax_confidence, ax_total] = plt.subplots(figsize=(8, 5), sharex=True, nrows=4)
	
	#Choose colors
	cmap=cm.coolwarm #cm.coolwarm #cm.bwr #cm.RdBu_r
	
	#Choose color thresholds
	elev_min=distances.min()
	elev_max=distances.max()
	mid_val=distances.median()
	
	#Helper dictionary
	ax = {"Market": ax_market, "Economic": ax_economic, "Confidence": ax_confidence, "Total": ax_total}

	#Plot colorbars
	for col in distances.columns:
		cax = ax[col].imshow(distances[col][np.newaxis, :], aspect='auto', interpolation='nearest', cmap=cmap, clim=(elev_min[col], elev_max[col]), norm=MidpointNormalize(midpoint=mid_val[col],vmin=elev_min[col], vmax=elev_max[col]))
		ax[col].set_ylabel(col+' Dist', fontdict={'fontsize': 9})
		ax[col].axes.get_yaxis().set_ticks([])
			
	#Set the ticks to corresponding dates (formatted in %y i.e. 'yy') on the bottom graph
	ticks = np.arange(0,distances.shape[0],12) # 1 tick every 12m
	ticklabels = [distances.index[int(i)].strftime("'%y") for i in ticks[ticks!=distances.shape[0]]] #Map number is previous np.array to corresponding date in distances.index
	ax_total.set_xticks(ticks)
	ax_total.set_xticklabels(ticklabels, fontdict={'fontsize': 9})
	
	#Add colorbar and place it to the right
	cbaxes = fig.add_axes([0.92, 0.2, 0.02, 0.58]) 
	cbar = fig.colorbar(mappable=cax, cax = cbaxes, ticks=[elev_min["Total"], mid_val["Total"], elev_max["Total"]])
	cbar.ax.set_yticklabels(['More similar', '', 'Less similar'], rotation='vertical', fontdict={'fontsize': 9})

	#Set graph title
	ax_market.set_title('FIGURE 2 - Distance from April 2018 Market Environment, Total and by Category', fontdict={'fontsize': 10})
	
	#Show graph
	plt.show()


if __name__ == "__main__": 
	
	#Define weights and categories
	weights, market, economic, confidence = define_weights_categories()
	
	#Get and transform data
	df_total=transform_data(LT_history=180)
	
	#Multiply by weights
	df_total=df_total.assign(**weights).mul(df_total)
    
    #Calculate distances between all dates and reference date
	distances = calc_distances(df_total, market, economic, confidence, reference_date='2018-04-01')
	
	#Map distances
	plot_graphs(distances)