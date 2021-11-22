import pandas as pd
import lightkurve as lk
import pandas as pd
import numpy as np
import pickle


def get_nb_of_rows(n_rows):
    df = pd.read_csv('raw_data/data.csv')
    df_w_nb_of_planet = df[['kepid','koi_disposition']].groupby('kepid').count().reset_index()
    kepids = df_w_nb_of_planet['kepid']
    return kepids[0:n_rows]


def get_data(n_rows):
    star_dict={}
    kepids = get_nb_of_rows(n_rows)
    for kepid in kepids:
        temp = lk.search_lightcurve(f'kplr{kepid}', mission="Kepler").download_all().stitch()
        time = temp[f'time'].value
        flux = temp[f'flux'].value
        star_dict[f'{kepid}time'] = time
        star_dict[f'{kepid}flux'] = flux
        with open('raw_data/exoplanet.pkl','wb') as f:
            pickle.dump(star_dict,f)

if __name__ == '__main__':
    print(get_data(n_rows=2))
