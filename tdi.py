'''
  | tdi.py: "Marketing Budget Allocation"
  | author: Dominik Leier
  | Requirements: python libs: numpy, scipy, matplotlib
  | Usage: python tdi.py past_sales_figures.csv number_of_bootstrap_realization
  | Output: a png file (tdi.png) and on-screen log
  
  '''
__version__ = "0.1.0"


from pylab import *
from scipy import *
import csv


def main(filename='past_sales_figures.csv',pprice=30,cprice=20,no_boot_strapping_realization=10000, verbose=False):

  p,adb,number_of_stores=read_data(filename)

  pmeans=np.array([(mean(p[0][0])+mean(p[0][1]))/2.,mean(p[0][2]),(mean(p[0][3])+mean(p[0][4]))/2.,mean(p[0][5])])*pprice
  p01=np.concatenate([p[0][0],p[0][1]])
  p34=np.concatenate([p[0][3],p[0][4]])
  pstds=np.array([std(p01),std(p[0][2]),std(p34),std(p[0][5])])*pprice
  
  cmeans=np.array([(mean(p[1][0])+mean(p[1][1]))/2.,(mean(p[1][2])+mean(p[1][3]))/2.,mean(p[1][4]),mean(p[1][5])])*cprice
  c01=np.concatenate([p[1][0],p[1][1]])
  c23=np.concatenate([p[1][2],p[1][3]])
  cstds=np.array([std(c01),std(c23),std(p[1][4]),std(p[1][5])])*cprice
 
  fig=plt.figure(1,figsize=(13, 4))
  plt.clf()
  plt.subplots_adjust(left=0.08, bottom=0.11, right=0.99, top=0.96, wspace=0.3, hspace=0.2)
  ax = plt.subplot(1,3,1)
  ax.axis([-1E5,1.1E6,0,2E5])
  
  xticklabels = getp(gca(), 'xticklabels')
  yticklabels = getp(gca(), 'yticklabels')
  setp(xticklabels, fontsize=10, weight='roman', family='cursive')
  setp(yticklabels, fontsize=10, weight='roman', family='cursive')
  plt.ylabel(r'$profit$ $[euro]$',fontsize=12)
  plt.xlabel(r'$ad$ $budget$ $[euro]$',fontsize=12)
  
  prop = matplotlib.font_manager.FontProperties(size=10)
  redarea = Rectangle( (0,0), 1,1, fc="red",alpha=0.2, linewidth=1)
  bluearea= Rectangle( (0,0), 1,1, fc="blue",alpha=0.2, linewidth=1)
  
  kl, = plot([-10000,-20000],'k-', linewidth=1)
  
  ax.legend([kl,redarea,bluearea], [r'$\mu\pm\sigma$',r'$Paul$', r'$Calvin$'],loc='upper left', ncol=1, shadow=False, fancybox=False, numpoints=1, prop=prop,labelspacing=-0.0,columnspacing=-0.5)

  
  ax.plot(adb,pmeans,'r-')
  ax.plot(adb,cmeans,'b-')
  ax.fill_between(adb,pmeans+pstds,pmeans-pstds,color='r',alpha=0.2)
  ax.fill_between(adb,cmeans+cstds,cmeans-cstds,color='b',alpha=0.2)



  p_slope,p_intercpt = np.polyfit(adb, pmeans, 1)
  if verbose==True:
    print "slope and intercept of lin. model fit: ", '%.2f' % p_slope, '%.1f' % p_intercpt
  t=linspace(0,1E6,1000)
  y=p_slope*t+p_intercpt
  
  RMSD=0
  for i in range(0,len(pmeans)):
    RMSD+=(pmeans[i]-(p_slope*adb[i]+p_intercpt))**2.
  RMSD/=len(pmeans)
  RMSD=sqrt(RMSD)
  cv=RMSD/(mean(pmeans))
  if verbose==True:
   print "CV(RMSD)=", '%.1e' % cv
  ax.text(5E5,4E4,"CV(RMSD)="+ str('%.1e' % cv),color='red')
  
  
  c_slope,c_intercpt = np.polyfit(adb, cmeans, 1)
  t=linspace(0,1E6,1000)
  y=c_slope*t+c_intercpt
  
  RMSD=0
  for i in range(0,len(cmeans)):
    RMSD+=(cmeans[i]-(c_slope*adb[i]+c_intercpt))**2.
  RMSD/=len(cmeans)
  RMSD=sqrt(RMSD)
  cv=RMSD/(mean(cmeans))
  if verbose==True:
    print "CV(RMSD)=",'%.1e' % cv
  ax.text(5E5,3E4,"CV(RMSD)="+ str('%.1e' % cv),color='blue')
  
  peslope,peintercpt = np.polyfit(adb, pstds, 1)
  t=linspace(0,1E6,1000)
  y=peslope*t+peintercpt+p_slope*t+p_intercpt
  #ax.plot(t,y,'r--')
  y=-peslope*t-peintercpt+p_slope*t+p_intercpt
  #ax.plot(t,y,'r--')
  
  RMSD=0
  for i in range(0,len(pstds)):
    RMSD+=(pstds[i]-(peslope*adb[i]+peintercpt))**2.
  RMSD/=len(pstds)
  RMSD=sqrt(RMSD)
  cv=RMSD/(mean(pstds))
  if verbose==True:
    print "CV(RMSD)=",'%.1e' % cv
  #ax.text(5E5,5E4,"CV(RMSD)="+ str('%.1e' % cv),color='red')
  
  
  ceslope,ceintercpt = np.polyfit(adb, cstds, 1)
  t=linspace(0,1E6,1000)
  y=ceslope*t+ceintercpt+c_slope*t+c_intercpt
  #ax.plot(t,y,'b--')
  y=-ceslope*t-ceintercpt+c_slope*t+c_intercpt
  #ax.plot(t,y,'b--')
  
  RMSD=0
  for i in range(0,len(cstds)):
    RMSD+=(cstds[i]-(ceslope*adb[i]+ceintercpt))**2.
  RMSD/=len(cstds)
  RMSD=sqrt(RMSD)
  cv=RMSD/(mean(cstds))
  if verbose==True:
    print "CV(RMSD)=",'%.1e' % cv
  #ax.text(5E5,2E4,"CV(RMSD)="+ str('%.1e' % cv),color='blue')
  
  adb_fraction=linspace(0,1,100)
  total_revenue=(c_slope*adb_fraction*1E6+c_intercpt)+(p_slope*(1.-adb_fraction)*1E6+p_intercpt)
  total_std=(sqrt((ceslope*adb_fraction*1E6+ceintercpt)**2.+((peslope*(1.-adb_fraction)*1E6+peintercpt))**2.))
  
  rel_revenue=((c_slope*adb_fraction*1E6+c_intercpt)+(p_slope*(1.-adb_fraction)*1E6+p_intercpt))/(sqrt((ceslope*adb_fraction*1E6+ceintercpt)**2.+((peslope*(1.-adb_fraction)*1E6+peintercpt))**2.))


  ax1 = plt.subplot(1,3,2)
  xticklabels = getp(gca(), 'xticklabels')
  yticklabels = getp(gca(), 'yticklabels')
  setp(xticklabels, fontsize=10, weight='roman', family='cursive')
  setp(yticklabels, fontsize=10, weight='roman', family='cursive')
  plt.ylabel(r'$total$ $profit$ $[euro]$',fontsize=12)
  plt.xlabel(r'$100\%$ $Paul$         $ad$ $fraction$         $100\%$ $Calvin$ ',fontsize=12)

  ax1.plot(adb_fraction,total_revenue,'k-')
  ax1.fill_between(adb_fraction,total_revenue+total_std,total_revenue-total_std,color='k',alpha=0.2)
  
  
  ax2 = plt.subplot(1,3,3)
  ax2.axis([0,1,16,31])
  xticklabels = getp(gca(), 'xticklabels')
  yticklabels = getp(gca(), 'yticklabels')
  setp(xticklabels, fontsize=10, weight='roman', family='cursive')
  setp(yticklabels, fontsize=10, weight='roman', family='cursive')
  plt.ylabel(r'$total$ $profit$ $/$ $profit$ $st.$ $dev.$',fontsize=12)
  plt.xlabel(r'$100\%$ $Paul$         $ad$ $fraction$         $100\%$ $Calvin$ ',fontsize=12)

  prop = matplotlib.font_manager.FontProperties(size=10)
  greyarea = Rectangle( (0,0), 1,1, fc="k",alpha=0.2, linewidth=1)
  kl, = plot([-10000,-20000],'k-', linewidth=1)
  kd, = plot([-10000,-20000],'k--', linewidth=1)
  kp, = plot([-10000,-20000],'k:', linewidth=1)

  ax1.legend([kl,greyarea,kd,kp], [r'$model-based$ $\mu$',r'$model-based$ $\sigma$', r'$random$ $(same$ $store)$ $\mu\pm\sigma$', r'$completely$ $random$ $\mu\pm\sigma$'],loc='upper right', ncol=1, shadow=False, fancybox=False, numpoints=1, prop=prop,labelspacing=-0.2,columnspacing=-0.5)


  ax2.plot(adb_fraction,rel_revenue,'k-')
  a=c_slope*1E6
  b=c_intercpt
  cc=p_slope*1E6
  d=p_intercpt
  e=ceslope*1E6
  f=ceintercpt
  g=peslope*1E6
  h=peintercpt
  rmax=(-b*e*f - cc*e*f - d*e*f + a*f**2 - cc*f**2 + a*g**2 + b*g**2 + d*g**2 +
        2*a*g*h + b*g*h - cc*g*h + d*g*h + a*h**2 - cc*h**2)/(
                                                            b*e**2 + cc*e**2 + d*e**2 - a*e*f + cc*e*f + a*g**2 + b*g**2 + d*g**2 +
                                                            a*g*h - cc*g*h)
  ax2.text(rmax,max(rel_revenue),"max @ "+str(round(rmax,3))+"\n"+"  p/c~("+str(round((1-rmax)/rmax,3))+")",fontsize=8)


  #random
  adb_fraction=linspace(0,1,100)

  for level_of_rndness in range(1,3):
    total_revenue_arr=[]
    total_revenue_rnd_mean=np.zeros(len(adb_fraction))
    m = 0.0 # in-loop mean
    s = 0.0 # in-loop stdev
    for l in range(0,no_boot_strapping_realization):
      total_revenue_rnd=[]
      
      if level_of_rndness==1:
        k=random.randint(0,number_of_stores)
        prandom=np.array([p[0][random.randint(0,2)][k],p[0][2][k],p[0][random.randint(3,5)][k],p[0][5][k]])*pprice
        crandom=np.array([p[1][random.randint(0,2)][k],p[1][random.randint(2,4)][k],p[1][4][k],p[1][5][k]])*cprice
      if level_of_rndness==2:
        prandom=np.array([p[0][random.randint(0,2)][random.randint(0,number_of_stores)],p[0][2][random.randint(0,number_of_stores)],p[0][random.randint(3,5)][random.randint(0,number_of_stores)],p[0][5][random.randint(0,number_of_stores)]])*pprice
        crandom=np.array([p[1][random.randint(0,2)][random.randint(0,number_of_stores)],p[1][random.randint(2,4)][random.randint(0,number_of_stores)],p[1][4][random.randint(0,number_of_stores)],p[1][5][random.randint(0,number_of_stores)]])*cprice


      for iadb in adb_fraction:
        total_revenue_rnd.append(piecewise_lin_interp(iadb*1E6,adb,crandom)+ piecewise_lin_interp((1.-iadb)*1E6,adb,prandom))
      total_revenue_rnd_mean+=total_revenue_rnd

      # for on-the-fly calculation of std
      tmp_m = m;
      m += (array(total_revenue_rnd) - tmp_m) / (l+1)
      s += (array(total_revenue_rnd) - tmp_m) * (array(total_revenue_rnd) - m)


    total_revenue_rnd_std=sqrt(s / (l-1))
    total_revenue_rnd_mean=m
    rel_revenue_rnd=total_revenue_rnd_mean/total_revenue_rnd_std

    if level_of_rndness==1:
      style='k--'
    if level_of_rndness==2:
      style='k:'
    ax1.plot(adb_fraction,total_revenue_rnd_mean,style)
    ax1.plot(adb_fraction,total_revenue_rnd_mean+total_revenue_rnd_std,style)
    ax1.plot(adb_fraction,total_revenue_rnd_mean-total_revenue_rnd_std,style)
    ax2.plot(adb_fraction,rel_revenue_rnd,style)

    adb_frac_at_max_rel_rev=[]
    for i in range(1,len(rel_revenue_rnd)):
      if (rel_revenue_rnd[i-1]<rel_revenue_rnd[i] and rel_revenue_rnd[i]>rel_revenue_rnd[i+1]):
        adb_frac_at_max_rel_rev.append(adb_fraction[i])
        if adb_fraction[i] < 0.9:
          ax2.text(adb_fraction[i],rel_revenue_rnd[i]-0.2,"max @ "+str(round(adb_fraction[i],3))+"\n"+"  p/c~("+str(round((1-adb_fraction[i])/adb_fraction[i],3))+")",fontsize=8)

    if verbose==True:
      if level_of_rndness==1:
        print "The max. profit / st. dev. are at ", adb_frac_at_max_rel_rev, " for random (row) sampling (same store) "
      if level_of_rndness==2:
        print "The max. profit / st. dev. are at ", adb_frac_at_max_rel_rev, " for random sampling (picking sales from diff. stores) "
    
    
  savefig('tdi.png')
  return



def piecewise_lin_interp(adb,adbrange,storeprofit):
  for i in range(0, len(adbrange)-1):
    if adb >= adbrange[i] and adb <= adbrange[i+1]:
      m=(storeprofit[i+1]-storeprofit[i])/(adbrange[i+1]-adbrange[i])
      b=storeprofit[i]-m*adbrange[i]
      profit=m*adb+b
      return profit
  return 0


def read_data(csv_file):
  the_file = open(csv_file, 'rU')
  reader = csv.reader(the_file, delimiter=';')
  product_name_arr=[]
  product_ad_budget=[]
  adb_array=[]
  
  product_col_id=[]
  
  p=np.ndarray(shape=(2,6,10000), dtype=float)
  
  for i, row in enumerate(reader):
    if i == 0:
      print("Read product ...")
      product_row=row
      for j in range(0,len(row)):
        if row[j]!='':
          if row[j] not in product_name_arr:
            product_name_arr.append(row[j])
      product_columns=[]
      iadb_arr=[]
      for k in product_name_arr:
        product_columns.append([])
        iadb_arr.append([])
    
    
    if i == 1:
      print("Read ad budget ...")
      adb_row=row
      for j in range(0,len(row)):
        if row[j]!='':
          if j!=0:
            if int(row[j]) not in adb_array:
              adb_array.append(int(row[j]))
            for k in range(0,len(product_name_arr)):
              if product_row[j]==product_name_arr[k]:
                product_columns[k].append(int(row[j]))
                iadb_arr[k].append(j)
      
      for ii in range(0,len(product_name_arr)):
        X=product_columns[ii]
        Y=iadb_arr[ii]
        product_col_id.append([x for (y,x) in sorted(zip(X,Y))])
    
    if i==2:
      print("Read data ...")

    
    if i>1:
      
      j=0
      for row in reader:
        for jj in range(0,len(product_col_id)):
          p[jj][0][j],p[jj][1][j],p[jj][2][j],p[jj][3][j],p[jj][4][j],p[jj][5][j]=list(float(row[kk]) for kk in product_col_id[jj])
        
        j+=1
  
  
  number_of_stores=j
  
  return p,adb_array,number_of_stores


if __name__ == "__main__":
  if len(sys.argv) < 3 or len(sys.argv) > 3:
    print "type: python tdi.py <file.csv> <number_realizations>"
    print "e.g. python tdi.py past_sales_figures.csv 5000"
    print "continue with default: past_sales_figures.csv 5000"
    main('past_sales_figures.csv',30,20,5000,False)
  else:
    filename=sys.argv[1]
    no_realizations=int(sys.argv[2])
    main(filename,30,20,no_realizations,False)

