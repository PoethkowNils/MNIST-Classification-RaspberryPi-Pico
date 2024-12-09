#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstdint> // For uint8_t

#include "model.h"
#include "model_settings.h"
#include "mnist_model_data.h"

// new inlcudes
#include "tensorflow/lite/micro/kernels/reshape.h"


Model::Model() :
    model(nullptr),
    interpreter(nullptr),
    input(nullptr),
    error_reporter(nullptr)
{
}

Model::~Model()
{
    if (interpreter != NULL) {
        delete interpreter;
        interpreter = NULL;
    }
    if (model != NULL) {
        delete model;
        model = NULL;
    }
    if (input != NULL) {
        delete input;
        input = NULL;
    }
    if (error_reporter != NULL) {
        delete error_reporter;
        error_reporter = NULL;
    }
}

int Model::setup() 
{
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(mnist_model_data); // Get model saved as C array from mnist_model_data.h ????
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return 0;
  }

  // set length of Resolver (Unfortunatly has to be done this way due to constexpr)
  // initial_model: 9, mobile_net: 13, squeeze_net: 12, lenet-5: 11
  static tflite::MicroMutableOpResolver<11> micro_op_resolver;

  // What operations do I need? Change this according to model!
  // Add Operations needed

  /* #### For Initial Model  And Optmized Architecture Model #### */
  // micro_op_resolver.AddAdd();
  // micro_op_resolver.AddMul();
  // micro_op_resolver.AddElu();
  // micro_op_resolver.AddSoftmax(); // Added from updated model
  // micro_op_resolver.AddFullyConnected();
  // micro_op_resolver.AddReshape();
  // micro_op_resolver.AddQuantize();
  // micro_op_resolver.AddDequantize();
  // micro_op_resolver.AddRelu() // Added for Optimized Architecture Model

  /* #### For mobile net #### */
  // micro_op_resolver.AddConv2D();            // Conv2D
  // micro_op_resolver.AddDepthwiseConv2D();   // DepthwiseConv2D
  // micro_op_resolver.AddRelu();              // ReLU activation
  // micro_op_resolver.AddSoftmax();           // Softmax activation
  // micro_op_resolver.AddAveragePool2D();     // GlobalAveragePooling2D
  // micro_op_resolver.AddReshape();           // Reshape layers
  // micro_op_resolver.AddAdd();               // BatchNormalization computations
  // micro_op_resolver.AddQuantize();
  // micro_op_resolver.AddDequantize();
  // micro_op_resolver.AddMean();
  // micro_op_resolver.AddShape();
  // micro_op_resolver.AddStridedSlice();
  // micro_op_resolver.AddPack();

  /* #### For Squeeze Net #### */
  // micro_op_resolver.AddMul();            // might not need this
  // micro_op_resolver.AddAdd();
  // micro_op_resolver.AddConv2D();
  // micro_op_resolver.AddRelu();           // Relu Activation for fire modules
  // micro_op_resolver.AddMaxPool2D();
  // micro_op_resolver.AddAveragePool2D();
  // micro_op_resolver.AddMean();
  // micro_op_resolver.AddFullyConnected();
  // micro_op_resolver.AddSoftmax();        // For Output layer
  // micro_op_resolver.AddConcatenation();  // Used in Fire modules
  // micro_op_resolver.AddQuantize();
  // micro_op_resolver.AddDequantize();

  /* #### For Lenet-5 #### */
  micro_op_resolver.AddMul();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddTanh();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();        // For Output layer
  micro_op_resolver.AddReshape(); 
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();
  
  static uint8_t tensor_arena[arena_size];  // allocate tensor_arena
  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, arena_size);
  interpreter = &static_interpreter;

  
  // Allocate tensor
  
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
      error_reporter->Report("AllocateTensors() failed");
      return -1;
  }

  // Get input tensor
  input = interpreter->input(0);
  if (!input) {
      error_reporter->Report("Failed to get input tensor.");
      return -1;
  }

  return 1;
}

uint8_t* Model::input_data() {
  if (input == nullptr) {
    return nullptr;
  }
  return input->data.uint8;
}

int Model::byte_size() {
  if (input == nullptr) {
    return 0;
  }
  return input->bytes;
}

int Model::predict()
{
  std::cout << "Invocation started" << std::endl;
  // Run invoke inference, if error, return -1
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Model invocation failed.");
      return -1;
  }

  std::cout << "Invocation finished" << std::endl;

  TfLiteTensor* output = interpreter->output(0);

  // Return an index of the output neuron, which has maximum probability. (for MNIST 10 output neurons)
  
  // Ouput the probability for every number
  int result = 0;
  for (int i = 0; i<10; i++) {
    std::cout << i << ": ";
    float prob = static_cast<float>(output->data.uint8[i]) / 255.0;
    std::cout << prob << std::endl;
    if (output->data.uint8[i] > output->data.uint8[result])
      result = i;
  }
  
  return result;
}