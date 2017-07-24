#ifndef SHADER_SRC_H
#define SHADER_SRC_H


#define SRC_STRING(...) #__VA_ARGS__

// pass_through.glsl
 // vertex shader 顶点着色器
 // vertex shader for a single quad
// work is performed in the operation specific texture shader

    const GLchar *sl_pass_through =SRC_STRING
    (   precision highp float;
        attribute vec3 pos;
        attribute vec2 tex;
        varying vec2   outTex;
        void main(void)
        {
            // just pass the position and texture coords
            gl_Position = vec4(pos, 1.0);
            outTex = tex;
        }
    );




    // 片段着色器 standalone.glsl
    // fragment shader that calculates the matrix product and renders each
    // element to the bytes representing a 32-bit IEEE754 floating point in
    // the output RGBA canvas.
    // readPixel is used to read the bytes.
    const GLchar *sl_standalone = SRC_STRING
    (
        precision highp float;
        varying vec2      outTex;	// texture coords of row/column to calculate
        uniform sampler2D A;		// texture with data from padded A
        uniform sampler2D B_t;		// texture with data from padded transpose of B
        uniform int       K;		// number of elements in shared dimension
        uniform int       N;		// number of columns in output
        uniform int       pad;		//
        uniform float     alpha; 	// coefficient to multiplication

        //#pragma glslify: dot_rowcol = require(./dot_rowcol)
        float dot_rowcol(float y, float x, sampler2D A, sampler2D B_t, int K) {
            float delta_t = 1./float(K);// space (on texture) between elements
            float sum = 0.;			// sum for this row/column pair
            float z = 0.5 * (4.0 * delta_t);// position for shared dimension on source textures

            for (int l=0 ; l<4096 ; ++l) {
                if(l >= K / 4) break;    // stop when we finish the row/column
                // l is in pixel space, so we divide by four

                // retrieve next four elements from each texture
                vec4 a_ik = texture2D(  A, vec2(z, y));
                vec4 b_kj = texture2D(B_t, vec2(z, x));

            // use `dot` to process four elements at a time
                sum +=  dot(a_ik, b_kj);
                z += (4.0 * delta_t);      // (z + 0.5)*delta
            }
            return sum;
        }
        //#pragma glslify: encode_float = require(../encode_float)
        vec4 encode_float(float val) {

            // TODO: correctly handle denormal numbers
            // http://www.2ality.com/2012/04/number-encoding.html
            float a = abs(val);                           // encode absolute value + sign
            float exp = floor(log2(a));                 // number of powers of 2
            float mant = pow(2.,log2(a)-exp) * pow(2.,23.);  // multiply to fill 24 bits (implied leading 1)
            float mant1 = floor(mant / 256. / 256.);    // first 8 bits of mantissa
            float mant2 = mod(floor(mant / 256.),256.); // second 8 bits
            float mant3 = mod(mant,256.);               // third 8 bits

            highp float sign = 128.-128.*(a/val);			// sign bit is 256 or 0
            //highp float e = (sign+exp+127.)/510.;		// exponent and sign
            highp float m1 = (mant1-(128.*(1.-mod(exp+127.,2.))))/255.; // handle leading bit
            highp float m2 = (mant2)/255.;				// middle part
            highp float m3 = (mant3+.5)/255.;			// scale to 0 - 255
            
            // 上面的e的计算存在误差，导致实际结果不对 /510 ( /2 /255) /2造成的0.5会被四舍五入，多出“1”
            int ee = int(sign+exp+127.)/2;

            return vec4(m3,m2,m1, float(ee)/255.);
        }

        void main(void) {

            // get the implied row and column from .y and .x of passed (output)
            // texture coordinate. These map directly to input texture space when
            // the relevant dimensions are the same.
            float row_t = outTex.y;
            float col_t = outTex.x;
            // sum row x col for the passed pixel
            float sum = alpha * dot_rowcol(row_t, col_t * float(N + pad)/float(N), A, B_t, K);
            
            if (sum == 0.) {
                gl_FragColor = vec4(0.,0.,0.,0.);
                return;
            }

            // output vec4 with bytes for an IEEE754 32-bit floating point number
            gl_FragColor = encode_float(sum);
            
        }
    );


    // standalone_c.glsl
    // fragment shader that calculates the matrix product (with additive 'C' term)
// and renders each element to the bytes representing a 32-bit IEEE754 floating
// point in the output RGBA canvas.
// readPixel is used to read the bytes.

   const GLchar *sl_standalone_c = SRC_STRING
    (
        precision highp float;

        varying vec2      outTex;	// texture coords of row/column to calculate
        uniform sampler2D A;		// texture with data from padded A
        uniform sampler2D B_t;		// texture with data from padded transpose of B
        uniform sampler2D C;		// texture with data from C
        uniform int       K;		// number of elements in shared dimension
        uniform int       N;		// number of columns in output
        uniform int       pad;		//
        uniform float     alpha; 	// coefficient to multiplication
        uniform float     beta; 	// coefficient to additive term

        // #pragma glslify: dot_rowcol = require(./dot_rowcol)
        float dot_rowcol(float y, float x, sampler2D A, sampler2D B_t, int K) {
            float delta_t = 1./float(K);// space (on texture) between elements
            float sum = 0.;			// sum for this row/column pair
            float z = 0.5 * (4.0 * delta_t);// position for shared dimension on source textures

            for (int l=0 ; l<4096 ; ++l) {
                if(l >= K / 4) break;    // stop when we finish the row/column
                // l is in pixel space, so we divide by four

                // retrieve next four elements from each texture
                vec4 a_ik = texture2D(  A, vec2(z, y));
                vec4 b_kj = texture2D(B_t, vec2(z, x));

            // use `dot` to process four elements at a time
                sum +=  dot(a_ik, b_kj);
                z += (4.0 * delta_t);      // (z + 0.5)*delta
            }
            return sum;
        }
        // #pragma glslify: encode_float = require(../encode_float)
        vec4 encode_float(float val) {

            // TODO: correctly handle denormal numbers
            // http://www.2ality.com/2012/04/number-encoding.html
            float a = abs(val);                           // encode absolute value + sign
            float exp = floor(log2(a));                 // number of powers of 2
            float mant = pow(2.,log2(a)-exp) * pow(2.,23.);  // multiply to fill 24 bits (implied leading 1)
            float mant1 = floor(mant / 256. / 256.);    // first 8 bits of mantissa
            float mant2 = mod(floor(mant / 256.),256.); // second 8 bits
            float mant3 = mod(mant,256.);               // third 8 bits

            highp float sign = 128.-128.*(a/val);			// sign bit is 256 or 0
            highp float e = (sign+exp+127.)/510.;		// exponent and sign
            highp float m1 = (mant1-(128.*(1.-mod(exp+127.,2.))))/255.; // handle leading bit
            highp float m2 = (mant2)/255.;				// middle part
            highp float m3 = (mant3+.5)/255.;			// scale to 0 - 255

            return vec4(m3,m2,m1,e);
        }
        // #pragma glslify: select_index = require(../select_index)
        float select_index(vec4 v, int index){
            float val;
            if (index == 0) {
                val = v.r;
            } else if(index == 1) {
                val = v.g;
            } else if(index == 2) {
                val = v.b;
            } else if(index == 3){
                val = v.a;
            } else {
                // should never be here
                val = 0.0;
            }

            return val;
        }

        void main(void) {

            // get the implied row and column from .y and .x of passed (output)
            // texture coordinate. These map directly to input texture space when
            // the relevant dimensions are the same.
            float row_t = outTex.y;
            float col_t = outTex.x;
            vec4 c_vec = texture2D(C, vec2(col_t, 0.5));

            // should be -0.5, but that subtly breaks at zero
            float col = col_t * float(N + pad); // index of first element in pixel (matrix space)
            int channel = int(mod(col, 4.0 ));
            float c = select_index(c_vec, channel);

            // sum row x col for the passed pixel
            float sum = alpha * dot_rowcol(row_t, col_t * float(N + pad)/float(N), A, B_t, K);
            sum += beta * c;

            if (sum == 0.) {
                gl_FragColor = vec4(0.,0.,0.,0.);
                return;
            }

            // output vec4 with bytes for an IEEE754 32-bit floating point number
            gl_FragColor = encode_float(sum);
        }
    );
// pipeline.glsh
// fragment shader that calculates the matrix product and writes each
// element to a pixel component in a floating point texture.
// the output RGBA canvas.
// readPixel is used to read the bytes.

  const GLchar *sl_pipeline = SRC_STRING
    (
        precision highp float;

        varying vec2      outTex;	// texture coords of row/column to calculate
        uniform sampler2D A;		// texture with data from padded A
        uniform sampler2D B_t;		// texture with data from padded transpose of B
        uniform int       K;		// number of elements in shared dimension
        uniform int       N;		// number of columns in output
        uniform int       pad;		//
        uniform float     alpha; 	// coefficient to multiplication

        float dot_rowcol(float y, float x, sampler2D A, sampler2D B_t, int K) {
            float delta_t = 1./float(K);// space (on texture) between elements
            float sum = 0.;			// sum for this row/column pair
            float z = 0.5 * (4.0 * delta_t);// position for shared dimension on source textures

            for (int l=0 ; l<4096 ; ++l) {
                if(l >= K / 4) break;    // stop when we finish the row/column
                // l is in pixel space, so we divide by four

                // retrieve next four elements from each texture
                vec4 a_ik = texture2D(  A, vec2(z, y));
                vec4 b_kj = texture2D(B_t, vec2(z, x));

            // use `dot` to process four elements at a time
                sum +=  dot(a_ik, b_kj);
                z += (4.0 * delta_t);      // (z + 0.5)*delta
            }
            return sum;
        }

        void main(void) {

            // get the implied row and column from .y and .x of passed (output)
            // texture coordinate. These map directly to input texture space when
            // the relevant dimensions are the same.
            float row_t = outTex.y;
            float col_t = outTex.x;

            vec4 sum_v = vec4(0.0, 0.0, 0.0, 0.0);
            float col = (col_t * float(N + pad) - 2.0); // index of first element in pixel (matrix space)
            sum_v.r = alpha * dot_rowcol(row_t, (col + 0.5)/float(N), A, B_t, K);
            // is last element in pixel past row length?
            if(pad > 0 && (col + 4.0) > float(N) ) {
                // compute elements in padded region
                if(pad < 3){
                    sum_v.g = alpha * dot_rowcol(row_t, (col + 1.5)/float(N), A, B_t, K);
                }
                if(pad < 2){
                    sum_v.b = alpha * dot_rowcol(row_t, (col + 2.5)/float(N), A, B_t, K);
                }
            } else {
                sum_v.g = alpha * dot_rowcol(row_t, (col + 1.5)/float(N), A, B_t, K);
                sum_v.b = alpha * dot_rowcol(row_t, (col + 2.5)/float(N), A, B_t, K);
                sum_v.a = alpha * dot_rowcol(row_t, (col + 3.5)/float(N), A, B_t, K);
            }


            gl_FragColor = sum_v;
        }
    ); 

//pipeline_c.glsl
// fragment shader that calculates the matrix product and writes each
// element to a pixel component in a floating point texture.
// the output RGBA canvas.
// readPixel is used to read the bytes
  const GLchar *sl_pipeline_c = SRC_STRING
    (
        precision highp float;

        varying vec2      outTex;	// texture coords of row/column to calculate
        uniform sampler2D A;		// texture with data from padded A
        uniform sampler2D B_t;		// texture with data from padded transpose of B
        uniform sampler2D C;		// texture with data from C
        uniform int       K;		// number of elements in shared dimension
        uniform int       N;		// number of columns in output
        uniform int       pad;		//
        uniform float     alpha; 	// coefficient to multiplication
        uniform float     beta; 	// coefficient to addition

        float dot_rowcol(float y, float x, sampler2D A, sampler2D B_t, int K) {
            float delta_t = 1./float(K);// space (on texture) between elements
            float sum = 0.;			// sum for this row/column pair
            float z = 0.5 * (4.0 * delta_t);// position for shared dimension on source textures

            for (int l=0 ; l<4096 ; ++l) {
                if(l >= K / 4) break;    // stop when we finish the row/column
                // l is in pixel space, so we divide by four

                // retrieve next four elements from each texture
                vec4 a_ik = texture2D(  A, vec2(z, y));
                vec4 b_kj = texture2D(B_t, vec2(z, x));

            // use `dot` to process four elements at a time
                sum +=  dot(a_ik, b_kj);
                z += (4.0 * delta_t);      // (z + 0.5)*delta
            }
            return sum;
        }

        void main(void) {

            // get the implied row and column from .y and .x of passed (output)
            // texture coordinate. These map directly to input texture space when
            // the relevant dimensions are the same.
            float row_t = outTex.y;
            float col_t = outTex.x;
            vec4 c_v = texture2D(C, vec2(col_t, 0.5));

            vec4 sum_v = vec4(0.0, 0.0, 0.0, 0.0);
            float col = (col_t * float(N + pad) - 2.0); // index of first element in pixel (matrix space)
            sum_v.r = alpha * dot_rowcol(row_t, (col + 0.5)/float(N), A, B_t, K);
            // in the padding region?
            if(pad > 0 && (col + 4.0) > float(N) ) {
                // pad
                if(pad < 3){
                    sum_v.g = alpha * dot_rowcol(row_t, (col + 1.5)/float(N), A, B_t, K);
                }
                if(pad < 2){
                    sum_v.b = alpha * dot_rowcol(row_t, (col + 2.5)/float(N), A, B_t, K);
                }
            } else {
                sum_v.g = alpha * dot_rowcol(row_t, (col + 1.5)/float(N), A, B_t, K);
                sum_v.b = alpha * dot_rowcol(row_t, (col + 2.5)/float(N), A, B_t, K);
                sum_v.a = alpha * dot_rowcol(row_t, (col + 3.5)/float(N), A, B_t, K);
            }


            gl_FragColor = sum_v + beta*c_v;
        }

    ); 
 #endif