async function main() {
  const brr = await import("./brr.mjs");
  const gpu = new brr.GPU();

  const N = 1024 * 1024 * 32;
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

  {
    // Ref CPU

    const t0 = performance.now();
    for (let i = 0; i < 10; ++i) {
      fn_ref(A, B);
    }
    const t1 = performance.now();

    console.log(`CPU: ${Math.round(t1 - t0) / 10}us per iter`);
  }

  {
    // GPU

    const t0 = performance.now();
    for (let i = 0; i < 10; ++i) {
      await fn(A, B);
    }
    await gpu.block();
    const t1 = performance.now();

    console.log(`GPU: ${Math.round(t1 - t0) / 10}us per iter`);
  }

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

main().catch((err) => console.error(err));
