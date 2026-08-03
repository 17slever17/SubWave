class PCMChunkProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const configuredMs = Number(options.processorOptions?.chunkMs || 40);
    this.chunkSamples = Math.max(
      128,
      Math.round(sampleRate * configuredMs / 1000),
    );
    this.buffer = new Float32Array(this.chunkSamples);
    this.offset = 0;
    this.sequence = 0;
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels?.length) {
      return true;
    }
    const frames = channels[0].length;
    for (let index = 0; index < frames; index += 1) {
      let mono = 0;
      for (const channel of channels) {
        mono += channel[index] || 0;
      }
      this.buffer[this.offset] = mono / channels.length;
      this.offset += 1;
      if (this.offset === this.buffer.length) {
        const payload = this.buffer;
        this.port.postMessage(
          {
            type: 'pcm',
            sequence: this.sequence,
            samples: payload.buffer,
          },
          [payload.buffer],
        );
        this.sequence = (this.sequence + 1) >>> 0;
        this.buffer = new Float32Array(this.chunkSamples);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-chunk-processor', PCMChunkProcessor);
