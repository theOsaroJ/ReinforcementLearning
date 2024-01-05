from new_gpmodel_class import GPModel

prior_path = '/scratch365/eosaro/Research/DeepRL/BigData/dataset/Prior.csv'
test_path = '/scratch365/eosaro/Research/DeepRL/BigData/dataset/Test.csv'
final_prior_path = '/scratch365/eosaro/Research/DeepRL/BigData/dataset/results/final_prior.csv'
batch_size = 1000

model = GPModel(prior_path, test_path, batch_size)
#idx_list = [80, 432, 384, 293, 11, 27, 440, 272, 156, 423, 290, 287, 113, 164, 199, 184, 378, 202, 1, 291, 118, 269, 385, 215, 377, 190, 296, 44, 154, 400, 92, 129, 78, 99, 420, 472, 266, 40, 81, 307, 43, 289, 273, 448, 153, 314, 404, 444, 280, 419, 45, 249, 218, 429, 22, 239, 456, 317, 376, 231, 69, 277, 396, 430, 308, 42, 373, 210, 200, 76, 220, 366, 298, 149, 248, 135, 319, 454, 107, 82, 407, 125, 161, 343, 180, 134, 336, 83, 478, 13, 458, 281, 315, 339, 449, 90, 398, 262, 242, 186, 6, 386, 408, 16, 21, 0, 57, 64, 443, 187, 412, 51, 211, 371, 88, 8, 105, 286, 219, 475, 227, 345, 36, 338, 256, 63, 254, 60, 121, 136, 77, 255, 75, 142, 318, 68, 188, 224, 457, 350, 237, 354, 323, 301, 34, 251, 48, 321, 126, 340, 41, 208, 344, 214, 189, 228, 143, 66, 124, 152, 194]

model.add_data_list(idx_list)
model.update_model(None)
r2 = model.calculate_r2()
print(r2)

model.write_final_prior(final_prior_path)
