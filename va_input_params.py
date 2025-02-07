         
LAYERS = [{'relax' : 'tests.dump/main.0',
          'defect' : 'tests.dump/M_TST_1',
          'radius' : 2,
          'smoothing level' : 13, 
          'smoothing_level_training' : 13, 
          'cutoff radius' : 3, 
             'iteraciones':1 ,
              'bOvitoModifiers':True, #Si esta desactivado por defecto se utilizara MultiSOM
            'columns_train':[0,1]#surface,vecinos,norma menor,norma mayor
            ,'strees': [1,1,1],
            'cutoff cluster':1.5,
            'divisions_of_cluster':2#1  , 2 o 3
            ,'radius_training':3

          }]
