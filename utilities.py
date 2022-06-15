import os
import csv

class Utilities():
    def write_csv(self, model, csv_data):
        header = ['Class_ID', 'Name', 'Sample_Num', 'Flag', 'Gender']

        with open('./datasets/' + model['_id'] + '/identity_meta.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write the data
            for row in csv_data:
                writer.writerow(row)

    def write_model_dataset(self, files, model):

        csv_data = []

        for [i, celeb] in enumerate(files):
            celeb_id = 'n' + str(i)
            csv_data.append([celeb_id, celeb, 14, 1, 'm'])
            train_dataset_path = './datasets/' + model['_id'] + '/base/train/' + celeb_id
            test_dataset_path = './datasets/' + model['_id'] + '/base/test/' + celeb_id
            os.makedirs(train_dataset_path)
            os.makedirs(test_dataset_path)

            for [i, file] in enumerate(files[celeb]):
                with open(os.path.join(train_dataset_path, file['name']), 'wb') as fp:
                    fp.write(file['data'])

        self.write_csv(model, csv_data)