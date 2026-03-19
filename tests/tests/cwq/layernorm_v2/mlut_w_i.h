require_extension(EXT_MATRIX);
reg_t md = sext_xlen(insn.md() - P.MU.NUM_MR);
reg_t ms1 = sext_xlen(insn.ms1() - P.MU.NUM_MR);
reg_t uimm3 = insn.uimm3();
reg_t lut_type = insn.lut_type();
// reg_t tile_m = P.MU.tile_m;
reg_t tile_n = P.MU.tile_n;

fprintf(stderr, "[DEBUG] mlut_w_i: md=%d, ms1=%d, uimm3=%d, lut_type=%d\n", md, ms1, uimm3, lut_type);
// static int32_t int_sqrt(int32_t n) {
//   if (n == 0) return 0;

//   int bits = 0;
//   for (int32_t x = n; x > 0; x /= 2)
//     bits++;

//   int32_t x_prev = 1 << ((bits + 1) / 2);

//   while (1) {
//     int32_t x_next = (x_prev + n / x_prev) / 2;
//     if (x_next >= x_prev) return x_prev;
//     x_prev = x_next;
//   };
// }

for (reg_t n = 0; n < tile_n; n++) {
  uint32_t val = P.MU.acc_reg[ms1][uimm3][n];
  float inv = 0.0f;
  uint32_t fixed_inv = 0 ;
  if (lut_type == 0) { // 1/x
    fprintf(stderr, "[DEBUG] TYPE0:mlut_w_i: 1/x for val=%d\n", val);
    if(val == 0) val = 1; // avoid div zero , val is treated as Q16.16
    float f_val = (float)val / (1 << 16); // Q16.16 to float, val is treated as Q16.16
    inv = 1.0f / f_val;
    fixed_inv = (uint32_t)(inv * (1 << 8)); // float to Q8.8
  } else if (lut_type == 1) { // 1/sqrt Q24.8
    fprintf(stderr, "[DEBUG] TYPE1:mlut_w_i: 1/sqrt for val=%d\n", val);
    if(val == 0) {
      inv = 3.0f; // define 1/sqrt(0) = 3.0
      fprintf(stderr, "[DEBUG] val is zero, set inv to %f\n", inv);
    } else{
      val = val; //actually val is in Q5.3 format, here it's set to Q8.0 for calculation

      int bits = 0;
      uint32_t inv_sqrt = 0;

      fprintf(stderr, "[DEBUG] val shifted to Q8.0 format: %d\n", val);
      for (uint32_t x = val; x > 0; x /= 2)
        bits++;

      uint32_t x_prev = 1 << (int)((bits + 1) / 2);
      fprintf(stderr, "[DEBUG] Number of bits in val: %d\n", bits);
      fprintf(stderr, "[DEBUG] Initial x_prev for Newton-Raphson: %d\n", x_prev);
      while (1) {
        uint32_t x_next = (x_prev + val / x_prev) / 2;
        fprintf(stderr, "[DEBUG]x_next: %d, x_prev: %d\n", x_next, x_prev);
        if (x_next >= x_prev){
          inv_sqrt = x_prev;
          break;
        } else{
          x_prev = x_next;        
        }
      };      
      float f_sqrt = (float)inv_sqrt;
      inv = 1.0f / f_sqrt;
      inv = inv * 2.8284271247461903f; // val is shifted left by 3 bits (from Q5.3 to Q8.0), so multiply by 2^(2/3)=2.8284271247461903
      fprintf(stderr, "[DEBUG]inv_sqrt: %d, f_sqrt: %f, inv: %f\n", inv_sqrt, f_sqrt, inv);
    }
    fixed_inv = (uint32_t)(inv * (1 << 24)); // float to Q8.24
    // fprintf("[DEBUG] fixed_inv = %u\n", fixed_inv);
    fprintf(stderr, "[DEBUG] fixed_inv=%u\n", fixed_inv);
  }
  P.MU.acc_reg[md][0][n] = fixed_inv;
}


P.MU.print_acc(md);