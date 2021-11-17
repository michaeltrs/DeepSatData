import argparse
import shapefile
from shapely import geometry
import pandas as pd
import os


def main():
    args = parser.parse_args()

    rpg_file = os.path.join(args.rpg_dir, 'PARCELLES_GRAPHIQUES')

    sf = shapefile.Reader(rpg_file)
    year = args.rpg_dir.split("-")[-1]
    # print(year)

    data = []
    for i in range(len(sf)):
        # if i == 100:
        #     break
        if i % 1e6 == 0:
            print('processing record %d of %d' % (i, len(sf)))
        s = sf.shape(i)
        rec = sf.record(i)
        parcel = geometry.Polygon(s.points)
        data.append([parcel, rec[2]])

    data = pd.DataFrame(data, columns=['geometry', 'CODE_CULTU'])

    print("num parcels in data file: %d" % data.shape[0])

    codecultu = data['CODE_CULTU'].drop_duplicates().tolist()
    codecultu = {code: i + 1 for i, code in enumerate(codecultu)}

    data['ground_truth'] = data['CODE_CULTU'].map(codecultu)
    del data['CODE_CULTU']
    data['crs'] = args.epsg
    data['year'] = year
    data = data[['ground_truth', 'crs', 'year', 'geometry']]

    savedir = os.path.join(os.path.dirname(rpg_file), 'DF')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    data.to_csv(os.path.join(savedir, os.path.basename(rpg_file) + '_DF.csv'), index=False)

    pd.DataFrame([[k, v] for k, v in codecultu.items()], columns=['CODE_CULTU', 'ground_truth']) \
        .to_csv(os.path.join(savedir, os.path.basename(rpg_file) + '_DF_codes.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract polygons and ground truths from RPG data')
    parser.add_argument('--rpg-dir', type=str, help='Path to RPG directory')
    parser.add_argument('--epsg', default='2154', type=str, help='EPSG coordinate system for RPG data')
    
    main()
