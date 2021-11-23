import pandas as pd
import lightkurve as lk
import pandas as pd
import numpy as np
import pickle


def get_nb_of_rows(start,stop,type):
    if type == 'exo':
        df = pd.read_csv('raw_data/data.csv')
        df_w_nb_of_planet = df[df['koi_disposition'] == 'CONFIRMED']
        df_w_nb_of_planet = df_w_nb_of_planet[['kepid','koi_disposition']].groupby('kepid').count().reset_index()
        kepids = df_w_nb_of_planet['kepid'][start : stop]
    elif type == 'non_exo':
        df = pd.read_csv('raw_data/non_exo_stars_kepid.csv')
        kepids = df['Kepler ID'][start:stop]
    return kepids


def get_data():
    star_dict={}
    # kepids = get_nb_of_rows(start,stop)
    # for kepid in kepids:
    temp = lk.search_lightcurve(f'kplr{kepid}', mission="Kepler", quarter=[1,2,3,4,5])
    if len(temp) >= 5:
        temp = temp.download_all().stitch()
        time = temp[f'time'].value
        flux = temp[f'flux'].value
        star_dict[f'{kepid}time'] = time
        star_dict[f'{kepid}flux'] = flux
    return star_dict

if __name__ == '__main__':
    # n_slices = 1
    # slices_size = 1
    # origin = 1500
    start = 518
    stop = 1000
    type = 'non_exo'
    kepids = get_nb_of_rows(start,stop,type)
    for kepid in kepids:
        temp = get_data()
        if type == 'exo':
            with open(f'raw_data/exo/exoplanet{kepid}.pkl','wb') as f:
                pickle.dump(temp,f)
        else:
            with open(f'raw_data/non_exo/exoplanet{kepid}.pkl','wb') as f:
                pickle.dump(temp,f)

    # for i in range(origin,origin + n_slices):
    #     temp = get_data(slices_size*(i-1),slices_size*i)
    #     with open(f'raw_data/exoplanet{i}.pkl','wb') as f:
    #         pickle.dump(temp,f)
