from deepsparse import compile_model
from sparsezoo.models import classification


for batch_size in [1, 2, 4, 64]:
	# Download model and compile as optimized executable for your machine
	model = classification.resnet_50()
	engine = compile_model(model, batch_size=batch_size)

	# Fetch sample input and predict output using engine
	inputs = model.data_inputs.sample_batch(batch_size=batch_size)
	for i in range(3):
		outputs, inference_time = engine.timed_run(inputs)

		print("batch_size", batch_size)
		print("inference_time", inference_time)
	print("--")
