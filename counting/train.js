"use strict";
require('es6-shim');

var convnetjs = require("convnetjs");
var fs = require('fs');
var csv = require('csv-parser');
var getPixels = require("get-pixels");

//// PARAMETERS
var N_TRAIN = 90;
var ITER = 100;
var MAX_POINTS = 10;
var image_dimension = 10;
var image_channels = 1;

// define layers
var layer_defs = [];

layer_defs.push({type:'input', out_sx:image_dimension, out_sy:image_dimension, out_depth:image_channels});
layer_defs.push({type:'conv', sx:5, filters:50, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:2, stride:2});
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:2, stride:2});
// layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
// layer_defs.push({type:'pool', sx:2, stride:2});
// layer_defs.push({type:'softmax', num_classes:MAX_POINTS});
layer_defs.push({type:'regression', num_neurons:1});

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


var formatValues = function(file, number) {

  var pathImage = "../data/counting_samples/" + file;
  var image;
  
  return new Promise(function(resolve, reject){
    // we may use https://github.com/gchudnov/inkjet for jpeg
    getPixels(pathImage, function(err, pixels) {
      if(err) {
        console.log(err);
        reject(err);
      }
      image = pixels;
      var imageData = [];
      for (var i=0; i< image.data.length; i+=4){
        imageData.push(image.data[i.toString()])
      }

      var x = new convnetjs.Vol(imageData);
      // target
      var y = [parseFloat(number)];

      resolve({x : x, y : y});
    })
  })
}


var trainP = [];
var testP = [];

var lossWindow = new Window(N_TRAIN);
var k = 0;
var square_error = 0.0




fs.createReadStream("../data/counting.csv")
  .pipe(csv({separator: ','}))
  .on('data', function(data) {
    if (k < N_TRAIN)
      trainP.push(formatValues(data.file, data.number));
    else
      testP.push(formatValues(data.file, data.number));
    k++;
  })
  .on("end", function(){
    console.log("Starting training");
    Promise.all(trainP).then(function(train){
      //train
      for(var iters=0; iters<ITER; iters++) {
        train.forEach(function(row){
          var stats = trainer.train(row.x, row.y);
          lossWindow.add(stats.loss);
        })
        if (iters % 10 === 0) 
          console.log("step ", iters, " on ", ITER, ' loss', lossWindow.get_average());
      }

      //test
      Promise.all(testP).then(function(test){
        test.forEach(function(v){
          var predicted = net.forward(v.x);
          var error = predicted.w[0] - v.y;
          square_error += error * error;
          console.log('Predicted', predicted.w[0], 'Expected', v.y);
        })
        console.log("MSE", square_error / test.length);
        console.log("Finished")
      });
    });
  });
