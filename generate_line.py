import os
import cv2
from options.generate_options import GenerateOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.util import save_image

opt = GenerateOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.continue_train = False

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

counter = 1

# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)    
    model.test()
    
    visuals = model.get_current_visuals()    
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    comb = cv2.hconcat([visuals["real_A"], visuals["fake_D"]])
    # for key, img in visuals.items(): 
    #     output_path = os.path.join(opt.output_dir) + str(counter).zfill(4) + "_" + key + ".jpg"
    #     save_image(img, output_path)
    output_path = os.path.join(opt.output_dir) + str(counter).zfill(4) + ".jpg"
    save_image(comb, output_path)
    counter += 1
    cv2.imshow("result", comb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()