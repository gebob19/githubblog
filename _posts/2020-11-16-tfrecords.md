---
title:  "Video TFRecords: How to Efficiently Load Video Data"
description: "In this brief tutorial/example, I explain  the best way to store videos in TfRecords for more efficient and faster model training in TensorFlow version 1 / 1.15.0 ."
categories: examples
---

Compared to images, loading video data is expensive due to the I/O bottleneck and increased decoding time. This reduces efficiency leading to significantly longer training times. Reading online, there are generally two solutions for data loading videos:

1. Decode the video and save its matrix as is  

 - With this approach, we improve the speed by preprocessing the decoding; however, we aren't compressing, so storing a few videos which total a couple MBs ends up requiring a few GBs; not very memory efficient. 

2. Store the frames of the video as images using a folder filesystem

 - With this approach, the I/O limitations are reduced by reading the images directly and we take advantage of compression algorithms like JPEG. However, it would also require a large folder re-organization which isn't optimal.

The solution I came up with and will share with you is to store the video as a list of encoded images using TFRecords. This significantly improves data loading throughput (by at least 2x) without incurring large memory costs (maintains the same size).

# Setup 

### Software 

This code is written in `Tensorflow 1.15.0`; it should also work with `Tensorflow 2`.

### Data format

For this tutorial we need a `.txt` file for train, validation and test which is formatted like the following:

    {mp4 file path} {label}

For example, one line would look like:

    videos/54838.mp4 1951

# Creating the TFRecords

First, we look at how we create a TFRecord example from a video example.

<script src="https://gist.github.com/gebob19/4c4bcc6c04f5fb329e8d3b7570c84d4b.js"></script>

Then we loop through our dataset and save each example into a TFRecord. 

<script src="https://gist.github.com/gebob19/47b2e4be6c486f0e0caa7b62fcc9bd86.js"></script>

# Reading the TFRecord 

The most difficult part was figuring out how to decode the sequential frame data.

With simple solutions not working, being unable to find online resources and on top of it all working in mysterious bytes I created the solution through brute force. The result was a magical TensorFlow while loop.  

<script src="https://gist.github.com/gebob19/d4b14798a7dce32e7c684f261d4662bf.js"></script>

# Conclusion

That's it! Now you know how to encode and decode video data efficiently using TFRecords, happy hacking! :) 

A repo containing the full code can be found [here](https://github.com/gebob19/TFRecords_4_videos)!

If you enjoyed this post, you may enjoy my other posts! If you want to stay up to date you can find me on my [Github](https://github.com/gebob19) or [Twitter](https://twitter.com/brennangebotys)

### Why I made this 
- Difficult to find resources which are compatible with `Tensorflow 1.15.0` (mostly because `Tensorflow 2.0` is out)
- Lack of quality resources on how to use TFRecords with video data 
- Imo this is the best way to data load video data using Tensorflow 
- With video processing being such a cool field I'm sure many others will find this information useful in future research! 
