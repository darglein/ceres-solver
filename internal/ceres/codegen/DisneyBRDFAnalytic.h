#pragma once
#include <Eigen/Core>
#include <cmath>
#include <limits>

#include "ceres/codegen/codegen_cost_function.h"
namespace test {
struct DisneyBRDFAnalytic : public ceres::SizedCostFunction<3, 10> {
  template <typename T>
  inline T lerp(const T& a, const T& b, const T& u) const {
    return a + u * (b - a);
  }

  template <typename Derived1, typename Derived2>
  typename Derived1::PlainObject lerp(const Eigen::MatrixBase<Derived1>& a,
                                      const Eigen::MatrixBase<Derived2>& b,
                                      typename Derived1::Scalar alpha) const {
    return (typename Derived1::Scalar(1) - alpha) * a + alpha * b;
  }

  template <typename T>
  inline T sqr(const T& x) const {
    return x * x;
  }
  using Mat3 = Eigen::Matrix3d;

  using Vec3 = Eigen::Vector3d;

  DisneyBRDFAnalytic() {
    C_ = Eigen::Vector3d(0.1, 0.2, 0.3);
    N_ = Eigen::Vector3d(-0.1, 0.5, 0.2).normalized();
    V_ = Eigen::Vector3d(0.5, -0.2, 0.9).normalized();
    L_ = Eigen::Vector3d(-0.3, 0.4, -0.3).normalized();
    X_ = Eigen::Vector3d(0.5, 0.7, -0.1).normalized();
    Y_ = Eigen::Vector3d(0.2, -0.2, -0.2).normalized();
  }

  Eigen::Vector3d C_, N_, V_, L_, X_, Y_;

  mutable double metallic;
  mutable double subsurface;
  mutable double specular;
  mutable double roughness;
  mutable double specularTint;
  mutable double anisotropic;
  mutable double sheen;
  mutable double sheenTint;
  mutable double clearcoat;
  mutable double clearcoatGloss;

  using T = double;

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const {
    using ceres::Ternary;
    auto params = parameters[0];
    metallic = params[0];
    subsurface = params[1];
    specular = params[2];
    roughness = params[3];
    specularTint = params[4];
    anisotropic = params[5];
    sheen = params[6];
    sheenTint = params[7];
    clearcoat = params[8];
    clearcoatGloss = params[9];

    using Vec2 = Eigen::Matrix<T, 2, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;

    Vec3 C, N, V, L, X, Y;

    C(0) = CERES_LOCAL_VARIABLE(T, C_(0));
    C(1) = CERES_LOCAL_VARIABLE(T, C_(1));
    C(2) = CERES_LOCAL_VARIABLE(T, C_(2));

    N(0) = CERES_LOCAL_VARIABLE(T, N_(0));
    N(1) = CERES_LOCAL_VARIABLE(T, N_(1));
    N(2) = CERES_LOCAL_VARIABLE(T, N_(2));

    V(0) = CERES_LOCAL_VARIABLE(T, V_(0));
    V(1) = CERES_LOCAL_VARIABLE(T, V_(1));
    V(2) = CERES_LOCAL_VARIABLE(T, V_(2));

    L(0) = CERES_LOCAL_VARIABLE(T, L_(0));
    L(1) = CERES_LOCAL_VARIABLE(T, L_(1));
    L(2) = CERES_LOCAL_VARIABLE(T, L_(2));

    X(0) = CERES_LOCAL_VARIABLE(T, X_(0));
    X(1) = CERES_LOCAL_VARIABLE(T, X_(1));
    X(2) = CERES_LOCAL_VARIABLE(T, X_(2));

    Y(0) = CERES_LOCAL_VARIABLE(T, Y_(0));
    Y(1) = CERES_LOCAL_VARIABLE(T, Y_(1));
    Y(2) = CERES_LOCAL_VARIABLE(T, Y_(2));

    const T NdotL = N.dot(L);
    const T NdotV = N.dot(V);

    const Vec3 LpV = L + V;
    const Vec3 H = LpV / LpV.norm();

    const T NdotH = N.dot(H);
    const T LdotH = L.dot(H);

    const T HdotX = H.dot(X);
    const T HdotY = H.dot(Y);

    const Vec3 Cdlin = C;
    const T Cdlum = T(0.3) * Cdlin[0] + T(0.6) * Cdlin[1] + T(0.1) * Cdlin[2];

    const Vec3 Ctint = Cdlin / Cdlum;

    const Vec3 Cspec0 = lerp(
        specular * T(0.08) * lerp(Vec3(T(1), T(1), T(1)), Ctint, specularTint),
        Cdlin,
        metallic);
    const Vec3 Csheen = lerp(Vec3(T(1), T(1), T(1)), Ctint, sheenTint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and mix in diffuse retro-reflection based on roughness
    const T FL = schlickFresnel(NdotL);
    const T FV = schlickFresnel(NdotV);
    const T Fd90 = T(0.5) + T(2) * LdotH * LdotH * roughness;
    const T Fd = lerp(T(1), Fd90, FL) * lerp(T(1), Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    const T Fss90 = LdotH * LdotH * roughness;
    const T Fss = lerp(T(1), Fss90, FL) * lerp(T(1), Fss90, FV);
    const T ss = T(1.25) * (Fss * (T(1) / (NdotL + NdotV) - T(0.5)) + T(0.5));

    // specular
    const T eps = T(0.001);
    const T aspct = aspect(anisotropic);
    const T axTemp = sqr(roughness) / aspct;
    const T ayTemp = sqr(roughness) * aspct;
    //    const T ax = axTemp < eps ? eps : axTemp;
    //    const T ay = ayTemp < eps ? eps : ayTemp;
    const T ax = Ternary(axTemp < eps, eps, axTemp);
    const T ay = Ternary(ayTemp < eps, eps, ayTemp);
    const T Ds = GTR2_aniso(NdotH, HdotX, HdotY, ax, ay);
    const T FH = schlickFresnel(LdotH);
    const Vec3 Fs = lerp(Cspec0, Vec3(T(1), T(1), T(1)), FH);
    const T roughg = sqr(roughness * T(0.5) + T(0.5));
    const T GGXNdotL = smithG_GGX(NdotL, roughg);
    const T GGXNdotV = smithG_GGX(NdotV, roughg);
    const T Gs = GGXNdotL * GGXNdotV;

    // sheen
    const Vec3 Fsheen = FH * sheen * Csheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    const T a = lerp(T(0.1), T(0.001), clearcoatGloss);
    const T Dr = GTR1(NdotH, a);
    const T Fr = lerp(T(0.04), T(1), FH);
    const T cGGXNdotL = smithG_GGX(NdotL, T(0.25));
    const T cGGXNdotV = smithG_GGX(NdotV, T(0.25));
    const T Gr = cGGXNdotL * cGGXNdotV;

    const Vec3 resultNoCosine =
        (T(1.0 / M_PI) * lerp(Fd, ss, subsurface) * Cdlin + Fsheen) *
            (T(1) - metallic) +
        Gs * Fs * Ds +
        Vec3(T(0.25), T(0.25), T(0.25)) * clearcoat * Gr * Fr * Dr;
    const Vec3 result = NdotL * resultNoCosine;
    residuals[0] = result(0);
    residuals[1] = result(1);
    residuals[2] = result(2);

    if (jacobians && jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 10, 3>> J(jacobians[0]);
      J.row(0) =
          dfdm(NdotL, Fd, ss, Cdlin, Fsheen, Gs, Ds, Ctint, FH).transpose();
      J.row(1) = dfdsub(NdotL, Cdlin, Fd, ss).transpose();
      J.row(2) = dfds(NdotL, Gs, Ds, FH, Ctint).transpose();
      J.row(3) = dfdr(NdotL,
                      FV,
                      FL,
                      LdotH,
                      Cdlin,
                      Fs,
                      Ds,
                      Gs,
                      NdotV,
                      NdotH,
                      HdotX,
                      HdotY,
                      ax,
                      ay,
                      aspct)
                     .transpose();
      J.row(4) = dfdt(NdotL, Gs, Ds, FH, Ctint).transpose();
      J.row(5) =
          dfda(NdotL, Gs, Fs, NdotH, HdotX, HdotY, ax, ay, aspct).transpose();
      J.row(6) = dfdsh(NdotL, FH, Csheen).transpose();
      J.row(7) = dfdsht(NdotL, FH, Ctint).transpose();
      J.row(8) = dfdc(NdotL, Gr, Fr, Dr).transpose();
      J.row(9) = dfdcg(NdotL, Gr, Fr, NdotH).transpose();
    }

    return true;
  }

  template <typename T>
  T schlickFresnel(const T& u) const {
    T m = T(1) - u;
    const T m2 = m * m;
    return m2 * m2 * m;  // (1-u)^5
  }

  template <typename T>
  T aspect(const T& anisotropic) const {
    return T(sqrt(T(1) - anisotropic * T(0.9)));
  }

  template <typename T>
  T smithG_GGX(const T& Ndotv, const T& alphaG) const {
    const T a = alphaG * alphaG;
    const T b = Ndotv * Ndotv;
    return T(1) / (Ndotv + T(sqrt(a + b - a * b)));
  }

  template <typename T>
  T GTR1(const T& NdotH, const T& a) const {
    T result = T(0);

    CERES_IF(a >= T(1)) { result = T(1 / M_PI); }
    CERES_ELSE {
      const T a2 = a * a;
      const T t = T(1) + (a2 - T(1)) * NdotH * NdotH;
      result = (a2 - T(1)) / (T(M_PI) * T(log(a2) * t));
    }
    CERES_ENDIF;
    return result;
  }

  template <typename T>
  T GTR2_aniso(const T& NdotH,
               const T& HdotX,
               const T& HdotY,
               const T& ax,
               const T& ay) const {
    return T(1) / (T(M_PI) * ax * ay *
                   sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH));
  }

  inline T GTR2_aniso(const T& NdotH,
                      const T& HdotX,
                      const T& HdotY,
                      const T& ax,
                      const T& ay) {
    return T(1) / (T(M_PI) * ax * ay *
                   sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH));
  }

  inline Vec3 dfdm(T NdotL,
                   T Fd,
                   T ss,
                   const Vec3& Cdlin,
                   const Vec3& Fsheen,
                   T Gs,
                   T Ds,
                   const Vec3& Ctint,
                   T FH) const {
    const Vec3 dFsDm = dFsdm(Ctint, Cdlin, FH);

    return NdotL *
           (-((T(1.0 / M_PI)) * lerp(Fd, ss, subsurface) * Cdlin + Fsheen) +
            Gs * Ds * dFsDm);
  }

  inline Vec3 dfdsub(T NdotL, const Vec3& Cdlin, T Fd, T ss) const {
    return (NdotL * T(1.0 / M_PI) * (T(1.0) - metallic) * (ss - Fd)) * Cdlin;
  }

  inline Vec3 dfds(T NdotL, T Gs, T Ds, T FH, const Vec3& Ctint) const {
    return (NdotL * Gs * Ds * (T(1.0) - FH) * (T(1.0) - metallic) * T(0.08)) *
           lerp(Vec3(T(1), T(1), T(1)), Ctint, specularTint);
  }

  inline Vec3 dfdr(T NdotL,
                   T FV,
                   T FL,
                   T LdotH,
                   const Vec3& Cdlin,
                   const Vec3& Fs,
                   T Ds,
                   T Gs,
                   T NdotV,
                   T NdotH,
                   T HdotX,
                   T HdotY,
                   T ax,
                   T ay,
                   T aspect) const {
    const T Fddr = dFddr(FV, FL, LdotH);
    const T Ssdr = dSsdr(FV, FL, LdotH, NdotL, NdotV);
    const T Gsdr = dGsDr(NdotV, NdotL);
    const T Dsdr = dDsdr(NdotH, HdotX, HdotY, ax, ay, aspect);

    return NdotL * ((T(1.0) - metallic) * T(1.0 / M_PI) *
                        lerp(Fddr, Ssdr, subsurface) * Cdlin +
                    Fs * (Gsdr * Ds + Gs * Dsdr));
  }

  inline Vec3 dfdt(T NdotL, T Gs, T Ds, T FH, const Vec3& Ctint) const {
    return (NdotL * Gs * Ds * (T(1.0) - FH) * specular * T(0.08) *
            (T(1.0) - metallic)) *
           (Ctint - Vec3(T(1.0), T(1.0), T(1.0)));
  }

  inline Vec3 dfda(T NdotL,
                   T Gs,
                   const Vec3& Fs,
                   T NdotH,
                   T HdotX,
                   T HdotY,
                   T ax,
                   T ay,
                   T aspect) const {
    const T dsda = dDsda(NdotH, HdotX, HdotY, ax, ay, aspect);
    return (NdotL * Gs * dsda) * Fs;
  }

  inline Vec3 dfdsh(T NdotL, T FH, const Vec3& Csheen) const {
    return (NdotL * (T(1.0) - metallic) * FH) * Csheen;
  }

  inline Vec3 dfdsht(T NdotL, T FH, const Vec3& Ctint) const {
    return (NdotL * (T(1.0) - metallic) * FH * sheen) *
           (Ctint - Vec3(T(1.0), T(1.0), T(1.0)));
  }

  inline Vec3 dfdc(T NdotL, T Gr, T Fr, T Dr) const {
    return Vec3::Ones() * NdotL * T(0.25) * Gr * Fr * Dr;
  }

  inline Vec3 dfdcg(T NdotL, T Gr, T Fr, T NdotH) const {
    const T a = lerp(T(0.1), T(0.001), clearcoatGloss);
    const T dGTR = dGTRdA(NdotH, a);
    const T dA = T(0.001 - 0.1);

    return NdotL * T(0.25) * clearcoat * Gr * Fr * dGTR * dA * Vec3::Ones();
  }

  inline Vec3 dFsdm(const Vec3& Ctint, const Vec3& Cdlin, T FH) const {
    return (Cdlin -
            specular * T(0.08) *
                lerp(Vec3(T(1.0), T(1.0), T(1.0)), Ctint, specularTint)) *
           (T(1.0) - FH);
  }

  inline T dGTRdA(T NdotH, T a) const {
    if (a >= T(1.0)) return T(0);

    const T a2 = a * a;
    const T NdotH2 = NdotH * NdotH;
    const T logA2 = log(a2);
    const T t = T(1.0) + (a2 - T(1.0)) * NdotH2;

    const T nominator =
        T(2.0) * a * (logA2 * t - (a2 - T(1.0)) * (t / a2 + logA2 * NdotH2));
    const T denominator = T(M_PI) * (logA2 * t) * (logA2 * t);
    return nominator / denominator;
  }

  inline T dFddr(T FV, T FL, T LdotH) const {
    const T TwoLdotH2 = T(2.0) * LdotH * LdotH;
    const T Fd90 = T(0.5) + TwoLdotH2 * roughness;
    const T dFd90Dr = TwoLdotH2;

    return dFd90Dr * (FV + FL + T(2.0) * FL * FV * (Fd90 - T(1.0)));
  }

  inline T dSsdr(T FV, T FL, T LdotH, T NdotL, T NdotV) const {
    const T LdotH2 = LdotH * LdotH;
    const T dFssdr = LdotH2 * (FV - T(2.0) * FL * FV + FL +
                               T(2.0) * LdotH2 * FL * FV * roughness);

    return T(1.25) * dFssdr * (T(1.0) / (NdotL + NdotV) - T(0.5));
  }

  T dGsDr(T NdotV, T NdotL) const {
    const T dAlpha = dAlphaDr();
    const T alpha = sqr(dAlpha);

    const T GGXNL = smithG_GGX(NdotL, alpha);
    const T GGXNV = smithG_GGX(NdotV, alpha);
    const T dGGXNL = dGGXdAlpha(NdotL, alpha);
    const T dGGXNV = dGGXdAlpha(NdotV, alpha);

    return dAlpha * (dGGXNL * GGXNV + GGXNL * dGGXNV);
  }

  T dGGXdAlpha(T NdotV, T alphaG) const {
    const T alphaG2 = alphaG * alphaG;
    const T NdotV2 = NdotV * NdotV;

    const T nominator = alphaG * (NdotV2 - T(1.0));

    const T sqrtDenom = sqrt(alphaG2 + NdotV2 - alphaG2 * NdotV2);
    const T denominator = sqr(NdotV + sqrtDenom) * sqrtDenom;
    return nominator / denominator;
  }

  T dAlphaDr() const { return T(0.5) * roughness + T(0.5); }

  T dDsdr(T NdotH, T HdotX, T HdotY, T ax, T ay, T aspect) const {
    const T HdotX2 = HdotX * HdotX;
    const T HdotY2 = HdotY * HdotY;
    const T NdotH2 = NdotH * NdotH;

    const T ax2 = ax * ax;
    const T ay2 = ay * ay;

    const T a2 = aspect * aspect;
    const T a4 = a2 * a2;

    const T k = HdotX2 / ax2 + HdotY2 / ay2 + NdotH2;
    const T k2 = k * k;
    const T k4 = k2 * k2;

    const T r2 = roughness * roughness;
    const T r3 = r2 * roughness;

    const T nominator =
        -T(4.0) * k *
        (k * r3 + T(2.0) * (-HdotX2 * a4 - HdotY2) / (a2 * roughness));
    const T denominator = T(M_PI) * ax2 * ay2 * k4;

    return nominator / denominator;
  }

  T dDsda(T NdotH, T HdotX, T HdotY, T ax, T ay, T aspect) const {
    const T dAspectdA = dAspectDa(aspect);
    const T dGTR2Aspect = dGTR2dAspect(NdotH, HdotX, HdotY, ax, ay, aspect);

    return dGTR2Aspect * dAspectdA;
  }

  T dGTR2dAspect(T NdotH, T HdotX, T HdotY, T ax, T ay, T aspect) const {
    const T HdotX2 = HdotX * HdotX;
    const T HdotY2 = HdotY * HdotY;
    const T NdotH2 = NdotH * NdotH;

    const T ax2 = ax * ax;
    const T ay2 = ay * ay;

    const T a2 = aspect * aspect;
    const T a3 = a2 * aspect;
    const T a4 = a2 * a2;

    const T k = HdotX2 / ax2 + HdotY2 / ay2 + NdotH2;
    const T k2 = k * k;
    const T k4 = k2 * k2;

    const T nominator = -T(4.0) * k * (HdotX2 * a4 - HdotY2) / a3;
    const T denominator = T(M_PI) * ax2 * ay2 * k4;

    return nominator / denominator;
  }

  T dAspectDa(T aspect) const { return -T(0.45) / aspect; }

  Vec3 dfddN(const Vec3& V,
             const Vec3& L,
             const T& NdotL,
             const T& NdotV,
             const T& Fd90,
             const T& Fss90,
             const T& Fss) const {
    const Vec3 dfdN = dFddN(V, L, NdotL, NdotV, Fd90);
    const Vec3 dfssdN = dssdN(V, L, NdotL, NdotV, Fss90, Fss);

    return T(1.0 / M_PI) * lerp(dfdN, dfssdN, subsurface);
  }

  Vec3 dFddN(const Vec3& V,
             const Vec3& L,
             const T& NdotL,
             const T& NdotV,
             const T& Fd90) const {
    const T dSchlickFresneldNdotL = dSchlickFresneldU(NdotL);
    const T dSchlickFresneldNdotV = dSchlickFresneldU(NdotV);

    const Vec3& dNdotLdN = L;
    const Vec3& dNdotVdN = V;

    const Vec3 dFLdN = dSchlickFresneldNdotL * -dNdotLdN;
    const Vec3 dFVdN = dSchlickFresneldNdotV * -dNdotVdN;

    return -dFLdN * (T(1) - Fd90) - dFVdN * (T(1) - Fd90);
  }

  Vec3 dssdN(const Vec3& V,
             const Vec3& L,
             const T& NdotL,
             const T& NdotV,
             const T& Fss90,
             const T& Fss) const {
    const Vec3 dFss = dFddN(V, L, NdotL, NdotV, Fss90);

    const T denominator = NdotL + NdotV;

    return T(1.25) * (dFss * (T(1) / denominator - T(0.5)) -
                      Fss / (denominator * denominator) * (L + V));
  }

  Vec3 dfsdN(const Vec3& V,
             const Vec3& L,
             const Vec3& H,
             const Mat3& dXdN,
             const Mat3& dYdN,
             const T& NdotL,
             const T& NdotV,
             const T& NdotH,
             const T& HdotX,
             const T& HdotY,
             const T& ax,
             const T& ay,
             const T& roughg,
             const T& GGXNdotL,
             const T& GGXNdotV,
             const T& Gs,
             const T& Ds) const {
    const Vec3 dGs = dGsdN(V, L, NdotL, NdotV, roughg, GGXNdotL, GGXNdotV);
    const Vec3 dDs = dDsdN(H, dXdN, dYdN, NdotH, HdotX, HdotY, ax, ay, Ds);

    return Gs * dDs + dGs * Ds;
  }

  Vec3 dGsdN(const Vec3& V,
             const Vec3& L,
             const T& NdotL,
             const T& NdotV,
             const T& roughg,
             const T& GGXNdotL,
             const T& GGXNdotV) const {
    const T dGGXdNdotL = dGGXdNdotM(NdotL, roughg);
    const T dGGXdNdotV = dGGXdNdotM(NdotV, roughg);

    const Vec3& dNdotVdN = V;
    const Vec3& dNdotLdN = L;

    return GGXNdotL * dGGXdNdotV * dNdotVdN + GGXNdotV * dGGXdNdotL * dNdotLdN;
  }

  Vec3 dDsdN(const Vec3& H,
             const Mat3& dXdN,
             const Mat3& dYdN,
             const T& NdotH,
             const T& HdotX,
             const T& HdotY,
             const T& ax,
             const T& ay,
             const T& GTR2) const {
    return dGTR2dN(H, dXdN, dYdN, NdotH, HdotX, HdotY, ax, ay, GTR2);
  }

  Vec3 dfcdN(const Vec3& V,
             const Vec3& L,
             const Vec3& H,
             const T& NdotL,
             const T& NdotV,
             const T& NdotH,
             const T& cGGXNdotL,
             const T& cGGXNdotV,
             const T& Fr,
             const T& Gr,
             const T& Dr,
             const T& a) const {
    const Vec3 dGr = dGrdN(V, L, NdotL, NdotV, cGGXNdotL, cGGXNdotV);
    const Vec3 dDr = dDrdN(H, NdotH, a, Dr);

    return T(0.25) * clearcoat * Fr * (dGr * Dr + Gr * dDr);
  }

  Vec3 dGrdN(const Vec3& V,
             const Vec3& L,
             const T& NdotL,
             const T& NdotV,
             const T& GGXNdotL,
             const T& GGXNdotV) const {
    const T dGGXdNdotL = dGGXdNdotM(NdotL, T(0.25));
    const T dGGXdNdotV = dGGXdNdotM(NdotV, T(0.25));

    const Vec3& dNdotVdN = V;
    const Vec3& dNdotLdN = L;

    return GGXNdotL * dGGXdNdotV * dNdotVdN + GGXNdotV * dGGXdNdotL * dNdotLdN;
  }

  Vec3 dDrdN(const Vec3& H, const T& NdotH, const T& a, const T& GTR1) const {
    const T dGTR1 = dGTR1dNdotH(NdotH, a, GTR1);
    const Vec3& NdotHdN = H;

    return dGTR1 * NdotHdN;
  }

  T dGGXdNdotM(const T& NdotV, const T& alphaG) const {
    const T a2 = alphaG * alphaG;
    const T NdotV2 = NdotV * NdotV;

    const T root = ceres::sqrt(a2 + NdotV2 - a2 * NdotV2);

    const T nominator = T(1) + NdotV * (T(1) - a2) / root;
    const T denominator = NdotV + root;

    return -nominator / (denominator * denominator);
  }

  T dGTR1dNdotH(const T& NdotH, const T& a, const T& GTR1) const {
    if (a >= T(1)) return T(0);

    const T a2 = a * a;
    const T b = a2 - T(1);
    const T c = T(1) + b * NdotH * NdotH;

    const T nominator = T(2) * b * b * NdotH;
    const T denominator = T(M_PI) * log(a2) * c * c;

    return -nominator / denominator;
  }

  T dSchlickFresneldU(const T& u) const {
    const T m = T(1) - u;
    const T m2 = m * m;
    return T(5) * m2 * m2;
  }

  Vec3 dGTR2dN(const Vec3& H,
               const Mat3& dXdN,
               const Mat3& dYdN,
               const T& NdotH,
               const T& HdotX,
               const T& HdotY,
               const T& ax,
               const T& ay,
               const T& GTR2) const {
    const T a = HdotX / ax;
    const T b = HdotY / ay;
    const T c = a * a + b * b + NdotH * NdotH;

    const Vec3& HdotXdX = H;
    const Vec3& HdotYdY = H;
    const Vec3& NdotHdN = H;

    const Vec3 HdotXdN = dXdN * HdotXdX;
    const Vec3 HdotYdN = dYdN * HdotYdY;

    const Vec3 nominator =
        T(4) * (a / ax * HdotXdN + b / ay * HdotYdN + NdotH * H);
    const T denominator = T(M_PI) * ax * ay * c * c * c;

    return -nominator / denominator;
  }
};

}  // namespace test
