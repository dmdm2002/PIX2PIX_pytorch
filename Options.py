class param(object):
    def __init__(self):
        # Path
        self.ROOT = 'Z:/2nd_paper'
        self.DATASET_PATH = f'{self.ROOT}/dataset/ND/original'
        self.OUTPUT_CKP = f'{self.ROOT}/PIX2PIX_temp'
        self.OUTPUT_SAMPLE = f'{self.ROOT}/PIX2PIX_temp/sampling'
        self.OUTPUT_TEST = f'{self.ROOT}/PIX2PIX_temp/test'
        self.OUTPUT_LOSS = ''
        self.CKP_LOAD = False

        # Data
        self.DATA_STYPE = ['A', 'B']
        self.SIZE = 256
        self.POOL_SIZE = 50

        # Train or Test
        self.EPOCH = 200
        self.LR = 2e-4
        self.B1 = 0.5
        self.B2 = 0.999
        self.LAMDA_PIXEL = 100
        self.LAMDA_ID = 0.5
        self.BATCHSZ = 2
        self.PERCEPTUAL = True

        # Handler
        # run_type 0 : train, 1 : test
        self.run_type = 0