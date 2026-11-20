# libecmcgbn: GMP-ECM 7.0.6 with stage 1 (CGBN) and stage 2 (small-prime
# NTTs, product trees, roots of F and G, batched over all curves) on an
# NVIDIA GPU. The same API as libecm (ecm.h, installed here as ecmcgbn.h),
# so a program links either libecm or libecmcgbn, not both.

# CUDA compute capability the kernels are built for (120: RTX PRO 4500 /
# 6000 Blackwell, GeForce RTX 50xx; 100: B200). Override with
# rpmbuild --define 'cuda_arch 100'.
%{!?cuda_arch:%global cuda_arch 120}
%{!?cuda_home:%global cuda_home /usr/local/cuda}
%{!?cgbn_include:%global cgbn_include /usr/local/include/cgbn}

# make check runs the GPU tests: it needs an NVIDIA GPU of compute
# capability cuda_arch (rpmbuild --without check on a machine without one)
%bcond check 1

# nvcc objects carry no build id / debuginfo in the form rpm expects
%global debug_package %{nil}

Name:           libecmcgbn
Version:        7.0.6
Release:        2%{?dist}
Summary:        Elliptic Curve Method for Integer Factorization on NVIDIA GPUs
License:        GPL-3.0-or-later AND LGPL-3.0-or-later
URL:            https://gitlab.inria.fr/zimmerma/ecm
Source0:        ecm-git-%{version}.tar.bz2
Patch0001:      0001-GPU-build-sm_100-sm_120-out-of-tree-.cu-compiles.patch
Patch0002:      0002-Stage-2-the-small-prime-NTT-products-on-the-GPU-gpu_.patch
Patch0003:      0003-Stage-2-conversions-to-and-from-residues-and-normali.patch
Patch0004:      0004-Stage-2-product-trees-and-multipoint-evaluation-on-t.patch
Patch0005:      0005-Stage-2-the-roots-of-F-and-G-on-the-GPU-gpu_roots.cu.patch
Patch0006:      0006-GPU-stage-1-N-may-fill-a-kernel-kernels-every-128-bi.patch
Patch0007:      0007-gpu-stage-2-of-several-curves-at-once-ECM_GPU_STAGE2.patch
Patch0008:      0008-gpu-stage-2-of-all-curves-at-once-on-the-GPU-batched.patch
Patch0009:      0009-gpu-stage-2-timed-by-the-wall-clock-README.gpu-stage.patch
Patch0010:      0010-GPU-code-in-the-shared-library-too-.cu-rule-without-.patch
# (0011 adds this spec file only)
Patch0012:      0012-ecm-with-the-GPU-NTT-NTT-stage-2-by-default-up-to-40.patch

ExclusiveArch:  x86_64

BuildRequires:  autoconf
BuildRequires:  automake
BuildRequires:  cgbn
BuildRequires:  cuda-cudart-devel-13-2
BuildRequires:  cuda-nvcc-13-2
BuildRequires:  gcc
BuildRequires:  gcc-c++
BuildRequires:  gmp-devel
BuildRequires:  libtool
BuildRequires:  m4
BuildRequires:  make

Requires:       %{name}-libs%{?_isa} = %{version}-%{release}

%description
GMP-ECM (programs and library for the elliptic curve method of integer
factorization) with its arithmetic on an NVIDIA GPU: stage 1 of many
curves at once with CGBN, and stage 2 (the small-prime NTT products, the
product trees and multipoint evaluation, the roots of F and G, for all
curves at once) in CUDA. The results are those of GMP-ECM. The program is
installed as ecmcgbn; use it with -gpu.

%package        libs
Summary:        Elliptic Curve Method library on NVIDIA GPUs
License:        LGPL-3.0-or-later

%description    libs
The GMP-ECM library with stage 1 and stage 2 on an NVIDIA GPU (libecmcgbn).

%package        devel
Summary:        Files for developing with libecmcgbn
License:        LGPL-3.0-or-later
Requires:       %{name}-libs%{?_isa} = %{version}-%{release}
Requires:       gmp-devel%{?_isa}

%description    devel
The header file ecmcgbn.h (the API of ecm.h) and the libecmcgbn.so link
for building programs with GPU GMP-ECM: include ecmcgbn.h and link with
the ecmcgbn library.

%prep
%autosetup -p1 -n ecm-git-%{version}

# Fix non-UTF-8 encodings (as gmp-ecm.spec)
iconv -f iso-8859-1 -t utf-8 -o AUTHORS.UTF-8 AUTHORS
touch -r AUTHORS AUTHORS.UTF-8
mv AUTHORS.UTF-8 AUTHORS
chmod 0644 AUTHORS

# Our own library name (soname libecmcgbn.so.1), so that it can be
# installed next to gmp-ecm-libs
sed -i -e 's/\blibecm\.la\b/libecmcgbn.la/g' \
       -e 's/\blibecm_la_/libecmcgbn_la_/g' Makefile.am

autoreconf -fi .


%build
export PATH=%{cuda_home}/bin:$PATH
%configure --disable-static --enable-shared --enable-openmp \
  --disable-gmp-cflags --disable-sse2 \
  --enable-gpu=%{cuda_arch} \
  --with-cuda=%{cuda_home} \
  --with-cgbn-include=%{cgbn_include} \
  CFLAGS='%{build_cflags} -Wa,--noexecstack' \
  LDFLAGS='%{build_ldflags} -Wl,-z,noexecstack -lgmp -lgomp'

# Eliminate hardcoded rpaths
sed -e 's|^hardcode_libdir_flag_spec=.*|hardcode_libdir_flag_spec=""|g' \
    -e 's|^runpath_var=LD_RUN_PATH|runpath_var=DIE_RPATH_DIE|g' \
    -i libtool

%make_build


%install
%make_install INSTALL="install -p"
rm -f $RPM_BUILD_ROOT%{_libdir}/*.la
mv $RPM_BUILD_ROOT%{_bindir}/ecm $RPM_BUILD_ROOT%{_bindir}/ecmcgbn
mv $RPM_BUILD_ROOT%{_includedir}/ecm.h $RPM_BUILD_ROOT%{_includedir}/ecmcgbn.h
mv $RPM_BUILD_ROOT%{_mandir}/man1/ecm.1 $RPM_BUILD_ROOT%{_mandir}/man1/ecmcgbn.1
# no debuginfo package (see above): strip here
strip --strip-unneeded $RPM_BUILD_ROOT%{_bindir}/ecmcgbn \
  $RPM_BUILD_ROOT%{_libdir}/libecmcgbn.so.1.*


%check
%if %{with check}
export LD_LIBRARY_PATH=$PWD/.libs
make check
%endif


%files
%doc README README.gpu
%license COPYING
%{_bindir}/ecmcgbn
%{_mandir}/man1/ecmcgbn.1*

%files libs
%doc AUTHORS ChangeLog NEWS TODO
%license COPYING.LIB
%{_libdir}/libecmcgbn.so.1
%{_libdir}/libecmcgbn.so.1.*

%files devel
%doc README.lib
%{_includedir}/ecmcgbn.h
%{_libdir}/libecmcgbn.so

%changelog
* Thu Nov 19 2026 Stefan Teleman <stefan.teleman@gmail.com> - 7.0.6-2
- Patch 0012: with the GPU NTT, ecmcgbn uses the NTT stage 2 (and the
  batched GPU stage 2) by default up to 4096 bits, not 1920

* Thu Nov 19 2026 Stefan Teleman <stefan.teleman@gmail.com> - 7.0.6-1
- GMP-ECM 7.0.6 (Fedora 41 gmp-ecm-7.0.6-2 sources) with stage 1 and
  stage 2 on NVIDIA GPUs (CGBN, CUDA), patches 0001-0010
- Renamed: libecmcgbn (soname libecmcgbn.so.1), ecmcgbn.h, ecmcgbn
