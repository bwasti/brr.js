function CHECK(str, fn) {
  const check = (cond) => {
    if (!cond) {
      throw Error(str);
    }
  }
  fn(check);
}

export class Memory {
  constructor(gpu, arrayType, numel) {
    this.gpu = gpu;
    this.numel = numel;
    this.arrayType = arrayType;
    this.buffer = null;
  }

  async createBuffer(usage) {
    const device = await this.gpu.device();
    // assume we want to write
    const size = this.numel * this.arrayType.BYTES_PER_ELEMENT;
    return device.createBuffer({size, usage});
  }

  async init() {
    if (this.buffer !== null) return;
    // kinda CoW
    this.buffer = await this.createBuffer(GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  }

  async read_gpu() {
    await this.init();
    if (!(this.buffer.usage & GPUBufferUsage.MAP_READ)) {
      if (this.buffer.usage == (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)) {
        throw Error(`Invalid buffer usage, did you call .read() on GPU outputs?`);
      }
      const device = await this.gpu.device();
      const usage = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
      const new_buffer = device.createBuffer({size:this.buffer.size, usage});
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.buffer, 0, new_buffer, 0, this.buffer.size);
      device.queue.submit([encoder.finish()]);
      this.buffer = new_buffer;
    }
    return this.buffer;
  }

  async storage_gpu() {
    await this.init();
    if (!(this.buffer.usage & GPUBufferUsage.STORAGE)) {
      const device = await this.gpu.device();
      let usage = GPUBufferUsage.STORAGE;
      if (this.buffer.usage & GPUBufferUsage.MAP_WRITE) {
        usage |= GPUBufferUsage.COPY_DST;
      } else if (this.buffer.usage & GPUBufferUsage.MAP_READ) {
        usage |= GPUBufferUsage.COPY_SRC;
      }
      const new_buffer = device.createBuffer({size:this.buffer.size, usage});
      const encoder = device.createCommandEncoder();
      if (this.buffer.usage & GPUBufferUsage.MAP_WRITE) {
        encoder.copyBufferToBuffer(this.buffer, 0, new_buffer, 0, this.buffer.size);
        device.queue.submit([encoder.finish()]);
      }
      this.buffer = new_buffer;
    }
    return this.buffer;
  }

  async write_gpu() {
    await this.init();
    if (!(this.buffer.usage & GPUBufferUsage.MAP_WRITE)) {
      const new_buffer = await this.createBuffer(GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC);      
      await this.read(async (arr) => {
        await new_buffer.mapAsync(GPUMapMode.WRITE);
        const new_arr = new this.arrayType(new_buffer.getMappedRange());
        new_arr.set(arr);
        await new_buffer.unmap();
        this.buffer = new_buffer;
      });
    }
    return this.buffer;
  }

  async exec_rw_callback(fn) {
    if (fn) {
      const out = await fn(new this.arrayType(this.buffer.getMappedRange()));
      await this.buffer.unmap();
      return out;
    }
    await this.buffer.unmap();
    return null;
  }

  async read(fn) {
    const rg = await this.read_gpu()
    await rg.mapAsync(GPUMapMode.READ);
    return await this.exec_rw_callback(fn);
  }
  async write(fn) {
    const wg = await this.write_gpu()
    await wg.mapAsync(GPUMapMode.WRITE);
    return await this.exec_rw_callback(fn);
  }
};

export class GPUFunction extends Function {
  constructor(code, options, device) {
    super();
    this.code = code;
    this.options = options;
    this.device = device;
    this.module = this.device.createShaderModule({code: this.code});
    this.pipeline = this.device.createComputePipeline({
      label: 'compute pipeline',
      layout: 'auto',
      compute: {
        module: this.module,
        entryPoint: this.options.name,
      },
    });
    return new Proxy(this, {
      apply: (target, thisArg, args) => target.run(...args)
    });
  }

  async run(...args) {
    let entries = [];
    for (let i = 0; i < args.length; ++i) {
      const buffer = await args[i].storage_gpu();
      entries.push({ binding: i, resource: { buffer: buffer }});
    }

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: entries
    });
    const encoder = this.device.createCommandEncoder({ label: 'compute builtin encoder' });
    const pass = encoder.beginComputePass({ label: 'compute builtin pass' });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...this.options.dispatch);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

};

export class GPU {
  constructor() {
    this.adapter_ = null;
    this.device_ = null;
    this.initialized = false;
  }
  async adapter() {
    await this.init();
    return this.adapter_;
  }
  async device() {
    await this.init();
    return this.device_;
  }
  async block() {
    await (await this.device()).queue.onSubmittedWorkDone();
  }
  async init() {
    if (this.initialized) {
      return;
    }
    this.adapter_ = await navigator.gpu?.requestAdapter();
    this.device_ = await this.adapter_?.requestDevice();
    if (!this.device_) {
      const e = new Error('Browser doesn\'t support WebGPU');
      console.error(e);
      throw e;
    }
    this.initialized = true;
  }

  alloc(arrayType, numel) {
    if (arrayType !== Float32Array) {
      throw Error(`Invalid Array type: '${arrayType}'`);
    }
    if (numel <= 0) {
      throw Error(`Invalid size: '${size}'`);
    }
    return new Memory(this, arrayType, numel);
  }

  async compile(args, kernel_fn, user_options = null) {
    // default options
    let options = {
      workgroup: [16],
      dispatch: [4],
      args: args,
      name: 'func',
    };
    if (user_options) {
      Object.assign(options, user_options);
    }

    CHECK('invalid workgroup, must be smaller than [256, 256, 64] and 256 total', (C) => {
      const workgroup = options.workgroup;
      C(workgroup.length < 4 && workgroup.length > 0);
      C(workgroup.reduce((a, b) => a * b) <= 256);
      if (workgroup.length === 1) {
        C(workgroup[0] <= 256);
      } else if (workgroup.length === 2) {
        C(workgroup[0] <= 256);
        C(workgroup[1] <= 256);
      } else if (workgroup.length === 3) {
        C(workgroup[0] <= 256);
        C(workgroup[1] <= 256);
        C(workgroup[2] <= 64);
      }
    });

    const prod = arr => arr.reduce((a, b) => a * b);
    const threads_per_workgroup = prod(options.workgroup);
    const num_workgroups = prod(options.dispatch);
    const num_threads = num_workgroups * threads_per_workgroup;

    const ctx = {
threads: num_threads,
         threads_per_workgroup: threads_per_workgroup,
         num_workgroups: num_workgroups,
         args: options.args.map((a)=>a.name),
    };
    const code_stub = kernel_fn(ctx);
    let bindings_stub = '';
    for (let i = 0;i < options.args.length; ++i) {
      const arg = options.args[i];
      bindings_stub += `  @group(0) @binding(${i}) var<storage, read_write> ${arg.name}: array<${arg.type}>;\n`
    }
    const code = `${bindings_stub}

  @compute @workgroup_size(${options.workgroup}) fn ${options.name}(
      @builtin(workgroup_id) workgroup_id : vec3<u32>,
      @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
      @builtin(global_invocation_id) global_invocation_id : vec3<u32>,
      @builtin(local_invocation_index) local_invocation_index: u32,
      @builtin(num_workgroups) num_workgroups: vec3<u32>
  ) {
    let workgroup_invocation_index: u32 = workgroup_id.x + workgroup_id.y * num_workgroups.x + workgroup_id.z * num_workgroups.x * num_workgroups.y;
    let global_invocation_index: u32 = ${threads_per_workgroup} * workgroup_invocation_index + local_invocation_index;

    ${code_stub}
  }
  `;
    return new GPUFunction(code, options, await this.device());
  }

};

//export function pointwise() {
//} 
