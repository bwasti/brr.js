async function mm(gpu) {
  const M = 32;
  const N = 32;
  const K = 32;
  const A = gpu.alloc(Float32Array, M * K);
  const B = gpu.alloc(Float32Array, K * N);
  const C = gpu.alloc(Float32Array, M * N);
  const sizes = gpu.alloc(Uint32Array, 3);
  await A.write((arr) => { arr.fill(1); })
  await B.write((arr) => { arr.fill(1); })
  await C.write((arr) => { arr.fill(0); })
  await sizes.write((arr) => {
	arr[0] = M;
	arr[1] = N;
	arr[2] = K;
  })
  const fn = await gpu.compile(
    [
      { name: "A", type: "f32" },
      { name: "B", type: "f32" },
      { name: "C", type: "f32", output: true },
      { name: "size", type: "u32" },
    ],
    (ctx) => `
    let M = size[0];
    let N = size[1];
    let K = size[2];
    for (var i = global_invocation_id.x; i < M;  i += num_workgroups.x * workgroup_size.x) {
      for (var j = global_invocation_id.y; j < N;  j += num_workgroups.y * workgroup_size.y) {
        for (var k : u32 = 0; k < K; k+=1) {
          C[i * N + j] += A[i * K + k] * B[k * N + j];
	}
      }
    }
  `,
    { workgroup: [4, 4], dispatch: [2, 2] },
  );

  await fn(A, B, C, sizes);

  await C.read((arr) => {
    console.log('matmul output', arr);
  });
}

async function main() {
  const brr = await import("./brr.js");
  const gpu = new brr.GPU();

  const N = 1024 * 1024 * 8;
  const fn = await gpu.compile(
    [
      { name: "A", type: "vec4<f32>" },
      { name: "B", type: "vec4<f32>" },
    ],
    (ctx) => `
    const delta = ${ctx.threads};
    var i = global_invocation_index;
    for (; i < ${N}; i += delta) {
      ${ctx.args[1]}[i] = ${ctx.args[0]}[i] * 2;
    }
  `,
    { workgroup: [256], dispatch: [256] },
  );

  const fn_ref = (A, B) => {
    for (let i = 0; i < N; ++i) {
      A[i] = B[i] * 2;
    }
  };

  const A = gpu.alloc(Float32Array, N);
  const B = gpu.alloc(Float32Array, N);

  const A_ref = new Float32Array(N);
  const B_ref = new Float32Array(N);

  await A.write((arr) => {
    for (let i = 0; i < N; ++i) {
      arr[i] = i;
      A_ref[i] = i;
    }
  });

  const cpu_complete = document.getElementById("cpu_complete");
  const cpu_time = document.getElementById("cpu_time");
  const gpu_complete = document.getElementById("gpu_complete");
  const gpu_time = document.getElementById("gpu_time");

  const iters = 100;
  const increment = 10;

  let cpu_total = 0;
  function doCPUIter(j) {
    if (j == iters) {
      doGPUIter(0);
      return;
    }
    const t0 = performance.now();
    for (let i = 0; i < increment; ++i) {
      fn_ref(A_ref, B_ref);
    }
    const t1 = performance.now();
    cpu_total += t1 - t0;
    cpu_time.textContent = `${
      Math.round((100 * cpu_total) / (j * increment)) / 100
    }us`;
    cpu_complete.textContent = `${Math.round(((j + 1) * 100) / iters)}%`;
    window.requestAnimationFrame(() => {
      doCPUIter(j + 1);
    }, 0);
  }

  let gpu_total = 0;
  async function doGPUIter(j) {
    if (j == iters) {
      return;
    }
    const t0 = performance.now();
    for (let i = 0; i < increment; ++i) {
      await fn(A, B);
    }
    await gpu.block();
    const t1 = performance.now();
    gpu_total += t1 - t0;
    gpu_time.textContent = `${
      Math.round((100 * gpu_total) / (j * increment)) / 100
    }us`;
    gpu_complete.textContent = `${Math.round(((j + 1) * 100) / iters)}%`;
    window.requestAnimationFrame(() => {
      doGPUIter(j + 1);
    }, 0);
  }

  doCPUIter(0);

  if (0) {
    await B.read((arr) => {
      for (let i = 0; i < Math.round(Math.sqrt(N)); ++i) {
        if (Math.abs(arr[i] - B_ref) > 0.0001) {
          console.assert(0);
          break;
        }
      }
      console.log(arr.slice(0, 10));
    });
  }

  // tests
  await mm(gpu);
}

main().catch((err) => console.error(err));
