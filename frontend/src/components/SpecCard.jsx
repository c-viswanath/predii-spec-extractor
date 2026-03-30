import React from 'react';
import { BookOpen, MapPin } from 'lucide-react';

const SpecCard = ({ spec, index }) => {
  return (
    <div className="glass-panel spec-card" style={{ animationDelay: `${index * 0.15}s` }}>
      <div className="spec-header">
        <span className="spec-type-badge">{spec.spec_type || 'Specification'}</span>
      </div>
      
      <h3 className="spec-component">{spec.component || 'Unknown Component'}</h3>
      
      <div className="spec-value-container">
        <span className="spec-value">{spec.value || '-'}</span>
        <span className="spec-unit">{spec.unit || ''}</span>
      </div>
      
      <div className="spec-context">
        {spec.context || 'Context is implicit.'}
      </div>
      
      <div className="spec-footer">
        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <MapPin size={14} style={{ color: 'var(--accent-blue)' }} /> 
          pg. {spec.source_page || '?'}
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <BookOpen size={14} style={{ color: 'var(--text-muted)' }} /> 
          Verified
        </span>
      </div>
    </div>
  );
};

export default SpecCard;
