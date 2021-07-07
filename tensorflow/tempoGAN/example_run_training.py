import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

#using density as input of generator, with tempoD
os.system('python tempoGAN.py learningRate 0.0000001 collapse_z 0 saveInterval 200 randSeed 42 out 0 trainingIters 400000 lambda 5.0 lambda2 -0.00001 discRuns 2 genRuns 2 alwaysSave 1 fromSim 1011 toSim 1011 outputInterval 200 genValiImg 0 dataDim 2 batchSize 16 useVelocities 0 useVorticities 0 gif 0 genModel gen_resnet discModel disc_binclass basePath ../2ddata_gan/ loadPath ../2ddata_sim/ lambda_t 1.0 lambda_t_l2 0.0 frameMax 240 data_fraction .1 adv_flag 0 dataAugmentation 0 rot 2 decayLR 1')

# os.system('python tempoGAN.py randSeed 42 out 0 trainingIters 200 lambda 5.0 lambda2 -0.00001 discRuns 2 genRuns 2 alwaysSave 1 fromSim 1000 toSim 1001 outputInterval 200 genValiImg 1 dataDim 2 batchSize 16 useVelocities 1 useVorticities 0 gif 0 genModel gen_resnet discModel disc_binclass basePath ../2ddata_gan/ loadPath ../2ddata_sim/ lambda_t 1.0 lambda_t_l2 0.0 frameMax 120 data_fraction 0.05 adv_flag 1 dataAugmentation 1 rot 2 decayLR 1')

"""
#using density and velocity as inputs of generator, with tempoD
os.system('python tempoGAN.py randSeed 42 out 0 trainingIters 40000 lambda 5.0 lambda2 -0.00001 discRuns 2 genRuns 2 alwaysSave 1 fromSim 1000 toSim 1001 outputInterval 200 genValiImg 1 dataDim 2 batchSize 16 useVelocities 1 useVorticities 0 gif 0 genModel gen_resnet discModel disc_binclass basePath ../2ddata_gan/ loadPath ../2ddata_sim/ lambda_t 1.0 lambda_t_l2 0.0 frameMax 120 data_fraction 0.05 adv_flag 1 dataAugmentation 1 rot 2 decayLR 1')

#using density, velocity and vorticity as inputs of generator, with tempoD
os.system('python tempoGAN.py randSeed 42 out 0 trainingIters 40000 lambda 5.0 lambda2 -0.00001 discRuns 2 genRuns 2 alwaysSave 1 fromSim 1000 toSim 1001 outputInterval 200 genValiImg 1 dataDim 2 batchSize 16 useVelocities 1 useVorticities 1 gif 0 genModel gen_resnet discModel disc_binclass basePath ../2ddata_gan/ loadPath ../2ddata_sim/ lambda_t 1.0 lambda_t_l2 0.0 frameMax 120 data_fraction 0.05 adv_flag 1 dataAugmentation 1 rot 2 decayLR 1')

#using density, velocity and vorticity as inputs of generator, l2 loss for temporal
os.system('python tempoGAN.py randSeed 42 out 0 trainingIters 40000 lambda 5.0 lambda2 -0.00001 discRuns 2 genRuns 2 alwaysSave 1 fromSim 1000 toSim 1001 outputInterval 200 genValiImg 1 dataDim 2 batchSize 16 useVelocities 1 useVorticities 1 gif 0 genModel gen_resnet discModel disc_binclass basePath ../2ddata_gan/ loadPath ../2ddata_sim/ lambda_t 0.0 lambda_t_l2 1.0 frameMax 120 data_fraction 0.05 adv_flag 1 dataAugmentation 1 rot 2 decayLR 1')
"""