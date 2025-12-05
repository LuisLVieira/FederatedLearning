from models import models_definition

model = models_definition.build_model('custom_layer4_fc_resnet18', num_classes = 4)

print(model)