c
c
      real function r1mach(i)
c
c  Single-precision machine constants
c
c  r1mach(1) = b**(emin-1), the smallest positive magnitude.
c
c  r1mach(2) = b**emax*(1 - b**(-t)), the largest magnitude.
c
c  r1mach(3) = b**(-t), the smallest relative spacing.
c
c  r1mach(4) = b**(1-t), the largest relative spacing.
c
c  r1mach(5) = log10(b)
c
c  To alter this function for a particular environment,
c  the desired set of data statements should be activated by
c  removing the c from column 1.
c  On rare machines a static statement may need to be added.
c  (But probably more systems prohibit it than require it.)
c
c  For IEEE-arithmetic machines (binary standard), the second
c  set of constants below should be appropriate.
c
c  Where possible, decimal, octal or hexadecimal constants are used
c  to specify the constants exactly.  Sometimes this requires using
c  equivalent integer arrays.  If your compiler uses half-word
c  integers by default (sometimes called integer*2), you may need to
c  change integer to integer*4 or otherwise instruct your compiler
c  to use full-word integers in the next 5 declarations.
c
      integer small(2)
      integer large(2)
      integer right(2)
      integer diver(2)
      integer log10(2)
      integer sc
c
      real rmach(5)
c
      equivalence (rmach(1),small(1))
      equivalence (rmach(2),large(1))
      equivalence (rmach(3),right(1))
      equivalence (rmach(4),diver(1))
      equivalence (rmach(5),log10(1))
c
c     machine constants for cdc cyber 205 and eta-10.
c
c     data small(1) / x'9000400000000000' /
c     data large(1) / x'6FFF7FFFFFFFFFFF' /
c     data right(1) / x'FFA3400000000000' /
c     data diver(1) / x'FFA4400000000000' /
c     data log10(1) / x'FFD04D104D427DE8' /, sc/987/
c
c     machine constants for ieee arithmetic machines, such as the at&t
c     3b series, motorola 68000 based machines (e.g. sun 3 and at&t
c     pc 7300), and 8087 based micros (e.g. ibm pc and at&t 6300).
c
       data small(1) /     8388608 /
       data large(1) /  2139095039 /
       data right(1) /   864026624 /
       data diver(1) /   872415232 /
       data log10(1) /  1050288283 /, sc/987/
c
c     machine constants for amdahl machines.
c
c      data small(1) /    1048576 /
c      data large(1) / 2147483647 /
c      data right(1) /  990904320 /
c      data diver(1) / 1007681536 /
c      data log10(1) / 1091781651 /, sc/987/
c
c     machine constants for the burroughs 1700 system.
c
c      data rmach(1) / z400800000 /
c      data rmach(2) / z5ffffffff /
c      data rmach(3) / z4e9800000 /
c      data rmach(4) / z4ea800000 /
c      data rmach(5) / z500e730e8 /, sc/987/
c
c     machine constants for the burroughs 5700/6700/7700 systems.
c
c      data rmach(1) / o1771000000000000 /
c      data rmach(2) / o0777777777777777 /
c      data rmach(3) / o1311000000000000 /
c      data rmach(4) / o1301000000000000 /
c      data rmach(5) / o1157163034761675 /, sc/987/
c
c     machine constants for ftn4 on the cdc 6000/7000 series.
c
c      data rmach(1) / 00564000000000000000b /
c      data rmach(2) / 37767777777777777776b /
c      data rmach(3) / 16414000000000000000b /
c      data rmach(4) / 16424000000000000000b /
c      data rmach(5) / 17164642023241175720b /, sc/987/
c
c     machine constants for ftn5 on the cdc 6000/7000 series.
c
c      data rmach(1) / o"00564000000000000000" /
c      data rmach(2) / o"37767777777777777776" /
c      data rmach(3) / o"16414000000000000000" /
c      data rmach(4) / o"16424000000000000000" /
c      data rmach(5) / o"17164642023241175720" /, sc/987/
c
c     machine constants for convex c-1.
c
c      data rmach(1) / '00800000'x /
c      data rmach(2) / '7fffffff'x /
c      data rmach(3) / '34800000'x /
c      data rmach(4) / '35000000'x /
c      data rmach(5) / '3f9a209b'x /, sc/987/
c
c     machine constants for the cray 1, xmp, 2, and 3.
c
c      data rmach(1) / 200034000000000000000b /
c      data rmach(2) / 577767777777777777776b /
c      data rmach(3) / 377224000000000000000b /
c      data rmach(4) / 377234000000000000000b /
c      data rmach(5) / 377774642023241175720b /, sc/987/
c
c     machine constants for the data general eclipse s/200.
c
c     note - it may be appropriate to include the following line -
c     static rmach(5)
c
c      data small/20k,0/,large/77777k,177777k/
c      data right/35420k,0/,diver/36020k,0/
c      data log10/40423k,42023k/, sc/987/
c
c     machine constants for the harris slash 6 and slash 7.
c
c      data small(1),small(2) / '20000000, '00000201 /
c      data large(1),large(2) / '37777777, '00000177 /
c      data right(1),right(2) / '20000000, '00000352 /
c      data diver(1),diver(2) / '20000000, '00000353 /
c      data log10(1),log10(2) / '23210115, '00000377 /, sc/987/
c
c     machine constants for the honeywell dps 8/70 series.
c
c      data rmach(1) / o402400000000 /
c      data rmach(2) / o376777777777 /
c      data rmach(3) / o714400000000 /
c      data rmach(4) / o716400000000 /
c      data rmach(5) / o776464202324 /, sc/987/
c
c     machine constants for the ibm 360/370 series,
c     the xerox sigma 5/7/9 and the sel systems 85/86.
c
c      data rmach(1) / z00100000 /
c      data rmach(2) / z7fffffff /
c      data rmach(3) / z3b100000 /
c      data rmach(4) / z3c100000 /
c      data rmach(5) / z41134413 /, sc/987/
c
c     machine constants for the interdata 8/32
c     with the unix system fortran 77 compiler.
c
c     for the interdata fortran vii compiler replace
c     the z's specifying hex constants with y's.
c
c      data rmach(1) / z'00100000' /
c      data rmach(2) / z'7effffff' /
c      data rmach(3) / z'3b100000' /
c      data rmach(4) / z'3c100000' /
c      data rmach(5) / z'41134413' /, sc/987/
c
c     machine constants for the pdp-10 (ka or ki processor).
c
c      data rmach(1) / "000400000000 /
c      data rmach(2) / "377777777777 /
c      data rmach(3) / "146400000000 /
c      data rmach(4) / "147400000000 /
c      data rmach(5) / "177464202324 /, sc/987/
c
c     machine constants for pdp-11 fortrans supporting
c     32-bit integers (expressed in integer and octal).
c
c      data small(1) /    8388608 /
c      data large(1) / 2147483647 /
c      data right(1) /  880803840 /
c      data diver(1) /  889192448 /
c      data log10(1) / 1067065499 /, sc/987/
c
c      data rmach(1) / o00040000000 /
c      data rmach(2) / o17777777777 /
c      data rmach(3) / o06440000000 /
c      data rmach(4) / o06500000000 /
c      data rmach(5) / o07746420233 /, sc/987/
c
c     machine constants for pdp-11 fortrans supporting
c     16-bit integers  (expressed in integer and octal).
c
c      data small(1),small(2) /   128,     0 /
c      data large(1),large(2) / 32767,    -1 /
c      data right(1),right(2) / 13440,     0 /
c      data diver(1),diver(2) / 13568,     0 /
c      data log10(1),log10(2) / 16282,  8347 /, sc/987/
c
c      data small(1),small(2) / o000200, o000000 /
c      data large(1),large(2) / o077777, o177777 /
c      data right(1),right(2) / o032200, o000000 /
c      data diver(1),diver(2) / o032400, o000000 /
c      data log10(1),log10(2) / o037632, o020233 /, sc/987/
c
c     machine constants for the sequent balance 8000.
c
c      data small(1) /  /
c      data large(1) / f7fffff /
c      data right(1) /  /
c      data diver(1) /  /
c      data log10(1) / e9a209b /, sc/987/
c
c     machine constants for the univac 1100 series.
c
c      data rmach(1) / o000400000000 /
c      data rmach(2) / o377777777777 /
c      data rmach(3) / o146400000000 /
c      data rmach(4) / o147400000000 /
c      data rmach(5) / o177464202324 /, sc/987/
c
c     machine constants for the vax unix f77 compiler.
c
c      data small(1) /       128 /
c      data large(1) /    -32769 /
c      data right(1) /     13440 /
c      data diver(1) /     13568 /
c      data log10(1) / 547045274 /, sc/987/
c
c     machine constants for the vax-11 with
c     fortran iv-plus compiler.
c
c      data rmach(1) / z00000080 /
c      data rmach(2) / zffff7fff /
c      data rmach(3) / z00003480 /
c      data rmach(4) / z00003500 /
c      data rmach(5) / z209b3f9a /, sc/987/
c
c     machine constants for vax/vms version 2.2.
c
c      data rmach(1) /       '80'x /
c      data rmach(2) / 'ffff7fff'x /
c      data rmach(3) /     '3480'x /
c      data rmach(4) /     '3500'x /
c      data rmach(5) / '209b3f9a'x /, sc/987/
c
c  ***  issue stop 778 if all data statements are commented...
      if (sc .ne. 987) stop 778
      if (i .lt. 1  .or.  i .gt. 5) goto 999
      r1mach = rmach(i)
      return
  999 write(*,1999) i
 1999 format(' r1mach - i out of bounds',i10)
      stop
      end

