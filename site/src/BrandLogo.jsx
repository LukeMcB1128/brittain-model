import { useId } from 'react';
import './brand-logo.css';

// Clip only the exterior background; preserve the supplied artwork inside.
export const logoOutline = 'M538 564 Q518 564 518 544 V361 Q518 348 528 340 L691 209 Q704 199 716 209 L787 265 V237 Q787 228 796 228 H825 Q835 228 835 238 V302 L881 338 Q890 345 890 360 V545 Q890 564 871 564 Z';
export default function BrandLogo({ className = '' }) {
  const maskId = useId();
  return (
    <svg className={`brand-logo ${className}`} viewBox="492 178 424 412" aria-hidden="true" focusable="false">
      <defs><clipPath id={maskId}><path d={logoOutline}/></clipPath></defs>
      <image href={`${import.meta.env.BASE_URL}brittain-code-logo.png`} width="1408" height="768" clipPath={`url(#${maskId})`}/>
    </svg>
  );
}
