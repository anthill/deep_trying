"use strict";

var convnetjs = require("convnetjs");
var fs = require('fs');
var csv = require('csv-parser');


//// PARAMETERS
var N_TRAIN = 400;
var ITER = 400;
var num_batches = 51; // 20 training batches, 1 test
var test_batch = 50;
var num_samples_per_batch = 1000;
var image_dimension = 100;
var image_channels = 1;
var use_validation_data = true;
var random_flip = true;
var random_position = true;

// define layers
var layer_defs = [];

layer_defs.push({type:'input', out_sx:image_dimension, out_sy:image_dimension, out_depth:image_channels});
layer_defs.push({type:'conv', sx:5, filters:50, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:2, stride:2});
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:2, stride:2});
// layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
// layer_defs.push({type:'pool', sx:2, stride:2});
layer_defs.push({type:'softmax', num_classes:10});

var getPixels = require("get-pixels")
getPixels("data/counting_samples/1.png", function(err, pixels) {
  if(err) {
    console.log("Bad image path")
    return
  }
  p=pixels
  console.log("got pixels", pixels.shape.slice())
})

// error window
var Window = function(size, minsize) {
    this.v = [];
    this.size = typeof(size)==='undefined' ? 100 : size;
    this.minsize = typeof(minsize)==='undefined' ? 10 : minsize;
    this.sum = 0;
  }

Window.prototype = {
    add: function(x) {
      this.v.push(x);
      this.sum += x;
      if(this.v.length>this.size) {
        var xold = this.v.shift();
        this.sum -= xold;
      }
    },
    get_average: function() {
      if(this.v.length < this.minsize) return -1;
      else return this.sum/this.v.length;
    },
    reset: function(x) {
      this.v = [];
      this.sum = 0;
    }
}

// create a net
var net = new convnetjs.Net();
net.makeLayers(layer_defs);
var trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:4, l2_decay:0.0001});


var formatValues = function(row) {
  var x = new convnetjs.Vol([
    parseFloat(row["LSTAT"]),
    parseFloat(row["RM"]), 
    parseFloat(row["DIS"]), 
    parseFloat(row["CRIM"]),
    parseFloat(row["NOX"]), 
    parseFloat(row["PTRATIO"]),
    parseFloat(row["TAX"]),
    parseFloat(row["AGE"]), 
    parseFloat(row["B"]),
    parseFloat(row["INDUS"]),
    parseFloat(row["CHAS"]), 
    parseFloat(row["RAD"]),
    parseFloat(row["ZN"])
    ]);
  // target
  var y = [parseFloat(row["MEDV"])]; // this must be a list

  return {x : x, y : y};
}


var train = [];
var test = [];

var lossWindow = new Window(N_TRAIN);
var k = 0;
var square_error = 0.0

fs.createReadStream("data/boston.csv")
  .pipe(csv({separator: ','}))
  .on('data', function(data) {
    var formated = formatValues(data);
    if (k < N_TRAIN){
      train.push(formated);
        k++;
      } else {
        test.push(formated);
      }
  })
  .on("end", function(){

    //train

    for(var iters=0; iters<ITER; iters++) {
      train.forEach(function(row){
        var stats = trainer.train(row.x, row.y);
        lossWindow.add(stats.loss);
      })
      if (iters % 10 === 0) 
        console.log("step ", iters, " on ", ITER, ' loss', lossWindow.get_average());
    }

    //testing

    test.forEach(function(v){
      var predicted = net.forward(v.x);
      var error = predicted.w[0] - v.y;
      square_error += error * error;
      console.log('Predicted', predicted.w[0], 'Expected', v.y);
    })
    console.log("MSE", square_error / test.length);
    console.log("Finished")
  });
